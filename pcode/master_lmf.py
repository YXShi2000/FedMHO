# -*- coding: utf-8 -*-
import os
import copy

import numpy as np
import torch
import torch.distributed as dist

import pcode.master_utils as master_utils
import pcode.create_coordinator as create_coordinator
import pcode.create_aggregator as create_aggregator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.cross_entropy as cross_entropy
from pcode.utils.early_stopping import EarlyStoppingTracker
from copy import deepcopy
from pcode.utils.str2top1 import str2top1


class Master(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        conf.n_participated = int(conf.n_clients * conf.participation_ratio)
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))
        self.conf.n_participated = conf.n_participated

        # create model as well as their corresponding state_dicts.
        # create server model.
        _, self.master_model = create_model.define_model(
            conf, to_consistent_model=False
        )
        # determine homogeneous or heterogeneous client models' arch.
        self.used_client_archs = set(
            [
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        )
        self.conf.used_client_archs = self.used_client_archs

        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")
        # create client models.
        self.client_models = dict(
            create_model.define_model(conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch
        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()  # Synchronization process
        self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        # if conf.data == "mnist":   # image size 28x28 -> 32 x 32
        #     pad = torch.nn.ZeroPad2d(padding=(2, 2, 2, 2))
        #     self.dataset["train"].data.data = pad(self.dataset["train"].data.data)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )
        conf.logger.log(f"Master initialized the local training data with workers.")

        # create val loader.
        # right now we just ignore the case of partitioned_by_user.
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None

        # create test loaders.
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        if conf.partitioned_by_user:
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id - 1,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:  # partitioned by labels
            # if conf.data == "mnist":  # image size 28x28 -> 32 x 32
            #     pad = torch.nn.ZeroPad2d(padding=(2, 2, 2, 2))
            #     self.dataset["test"].data.data = pad(self.dataset["test"].data.data)
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            self.test_loaders = [test_loader]

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        # define the aggregators.
        self.aggregator = create_aggregator.Aggregator(
            conf,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.dataset,
            test_loaders=self.test_loaders,
            clientid2arch=self.clientid2arch,
        )
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        conf.logger.log(f"Master initialized the aggregator/coordinator.\n")

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        try:
            self.num_arch = int(self.conf.n_clients / conf.arch_info["num_clients_per_model"] - 1)
            self.num_arch = 1 if self.num_arch == 0 else self.num_arch
        except:
            self.num_arch = 1

        self.semi_train_round = 1
        self.threshold = 0.55 if "mnist" in self.conf.data else 0.45
        self.class_book = np.array([0 for _ in range(10)])
        self.class_book_list = []
        self.prob_book = np.array([0.0 for _ in range(10)])
        self.semi_supervised_data_list, self.semi_supervised_label = [], []
        self.total_semi_supervised_data = [[] for _ in range(10)]
        self.synthesis_pool_each_class = 300

        # initialize variables for storing accuracy.
        self.arch_acc, self.arch_acc2, self.arch_acc_per_round, self.arch_acc_per_round2 = {}, {}, {}, {}
        self.avg_perfs_lst, self.avg_perfs_lst2 = [], []
        for _arch in conf.arch_info["worker"]:
            self.arch_acc[_arch], self.arch_acc2[_arch] = 0, 0
            self.arch_acc_per_round[_arch], self.arch_acc_per_round2[_arch] = [], []

        # save arguments to disk.
        conf.is_finished = False
        checkpoint.save_arguments(conf)

    def run(self):
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.
            selected_client_ids = self._random_select_clients()

            # detect early stopping.
            # reach target accuracy or performance plateaus for specific round.
            self._check_early_stopping()

            # init the activation tensor and broadcast to all clients (either start or stop).
            self._activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids)
            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                self._print_result()
                return

            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            )

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(flatten_local_models)
            print(self.arch_acc_per_round, "\n", self.arch_acc_per_round2)
            print(self.class_book, "\n", self.class_book_list, "\n", self.prob_book / self.conf.n_comm_rounds)

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        self._print_result()
        dist.barrier()
        self._finishing()

    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def _activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((3, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)
        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )
        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average",
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models):
        # uniformly averaged the model before the potential aggregation scheme.
        # used to check whether the clients are homogeneous.
        same_arch = (
                len(self.client_models) == 1
                and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]
        )

        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        if same_arch:  # homogeneous clients, the following codes will use fedavg_model
            fedavg_model = list(fedavg_models.values())[0]
        else:  # heterogenous clients, the following codes will use fedavg_models
            fedavg_model = None

        # (smarter) aggregate the model from clients.
        # note that: if conf.fl_aggregate["scheme"] == "federated_average",
        # then self.aggregator.aggregate_fn = None.(in create_aggregation.py line 35-39)
        # not federated_average
        if self.aggregator.aggregate_fn is not None:
            # evaluate the uniformly averaged model.
            # homogeneous clients.
            if fedavg_model is not None:
                performance = master_utils.get_avg_perf_on_dataloaders(
                    self.conf,
                    self.coordinator,
                    fedavg_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"fedag_test_loader",
                )
            # heterogenous clients.
            else:
                # evaluate each model prototype (_fedavg_models) on testset
                for _arch, _fedavg_model in fedavg_models.items():
                    if "vae" in _arch:
                        continue
                    print("-------------before kd-------------", _arch, "-------------")
                    tmp_performance = master_utils.get_avg_perf_on_dataloaders(
                        self.conf,
                        self.coordinator,
                        _fedavg_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"fedag_test_loader_{_arch}",
                    )
                    tmp_top1 = str2top1(str(tmp_performance))
                    self.arch_acc[_arch] = float(tmp_top1)
                # save average performance.
                tmp_sum_top1 = 0
                for key, value in self.arch_acc.items():
                    tmp_sum_top1 += value
                    self.arch_acc_per_round[key].append(value)
                self.avg_perfs_lst.append(tmp_sum_top1 / self.num_arch)

            # aggregate the local models.
            client_models, class_book, prob_book, total_semi_supervised_data = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                fedavg_model=fedavg_model,
                fedavg_models=fedavg_models,
                flatten_local_models=flatten_local_models,
                threshold=self.threshold,
                train_global_model=True if self.conf.graph.comm_round % self.semi_train_round == 0 else False,
                total_semi_supervised_data=self.total_semi_supervised_data,
            )
            self.total_semi_supervised_data = total_semi_supervised_data
            for i in range(10):
                if len(self.total_semi_supervised_data[i]) == self.synthesis_pool_each_class:
                    self.total_semi_supervised_data[i] = self.total_semi_supervised_data[i][0:int(self.synthesis_pool_each_class * 0.95)]
            # if self.conf.graph.comm_round % self.semi_train_round == 0:
            #     self.semi_supervised_data_list, self.semi_supervised_label = [], []
            #     self.class_book = np.array([0 for _ in range(10)])
            # else:
            #     self.semi_supervised_data_list += semi_supervised_data_list
            #     self.semi_supervised_label += semi_supervised_label
            #     self.class_book += class_book
            self.class_book_list.append(class_book)
            self.prob_book += prob_book
            self.threshold += 0.01
            if self.threshold >= 0.65 and "mnist" in self.conf.data:
                self.threshold = 0.65
            elif self.threshold >= 0.6 and "cifar" in self.conf.data:
                self.threshold = 0.6

            # # here the 'client_models' are updated in-place.
            # if same_arch:
            #     # here the 'master_model' is updated in-place only for 'same_arch' is True.
            #     self.master_model.load_state_dict(
            #         list(client_models.values())[0].state_dict()
            #     )

            for arch, _client_model in client_models.items():
                # update global model
                if "vae" not in arch:
                    self.master_model.load_state_dict(_client_model.state_dict())
                    print("------------------updated global model------------------")
                if not self.conf.one_shot:
                    try:
                        self.client_models[arch].load_state_dict(_client_model.state_dict())
                    except:
                        pass
        # federated_average
        else:
            # update self.master_model in place.
            if same_arch:
                self.master_model.load_state_dict(fedavg_model.state_dict())
            else:
                for _arch, _fedavg_model in fedavg_models.items():
                    print(">>>>>>>>", _arch, "begin testing accuracy...")
                    tmp_performance = master_utils.get_avg_perf_on_dataloaders(
                        self.conf,
                        self.coordinator,
                        _fedavg_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"fedag_test_loader_{_arch}",
                    )
                    tmp_top1 = str2top1(str(tmp_performance))
                    self.arch_acc[_arch] = float(tmp_top1)
                # save average performance.
                tmp_sum_top1 = 0
                for key, value in self.arch_acc.items():
                    tmp_sum_top1 += value
                    self.arch_acc_per_round[key].append(value)
                self.avg_perfs_lst.append(tmp_sum_top1 / self.num_arch)
            # update self.client_models in place.
            for arch, _fedavg_model in fedavg_models.items():
                self.client_models[arch].load_state_dict(_fedavg_model.state_dict())

        # evaluate the global model on the test data.
        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label=f"aggregated_test_loader",
        )
        # evaluate each model prototype on the test data.
        if self.aggregator.aggregate_fn is not None:
            for _arch, _client_model in self.client_models.items():
                if "vae" in _arch:
                    continue
                print("-------------after kd-------------", _arch, "-------------")
                tmp_performance = master_utils.get_avg_perf_on_dataloaders(
                    self.conf,
                    self.coordinator,
                    _client_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"fedag_test_loader_{_arch}",
                )
                tmp_top1 = str2top1(str(tmp_performance))
                self.arch_acc2[_arch] = float(tmp_top1)
            tmp_sum_top1 = 0
            for key, value in self.arch_acc2.items():
                tmp_sum_top1 += value
                self.arch_acc_per_round2[key].append(value)
            self.avg_perfs_lst2.append(tmp_sum_top1 / self.num_arch)

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")

    def _print_result(self):
        try:
            print("average performance of local models:")
            [print(f'the {i + 1}th round average performance on local models is {p}') for i, p in
             enumerate(self.avg_perfs_lst)]
            print("best:", max(self.avg_perfs_lst), self.avg_perfs_lst.index(max(self.avg_perfs_lst)) + 1,
                  "round")
            print("average performance of local models:")
            [print(f'the {i + 1}th round average performance on local models is {p}') for i, p in
             enumerate(self.avg_perfs_lst2)]
            print("best:", max(self.avg_perfs_lst2), self.avg_perfs_lst2.index(max(self.avg_perfs_lst2)) + 1,
                  "round")
            tmp_dict = {}
            for (k1, v1), (k2, v2) in zip(self.arch_acc_per_round.items(), self.arch_acc_per_round2.items()):
                tmp_lst = []
                for vv1, vv2 in zip(v1, v2):
                    tmp_lst.append(round(vv2 - vv1, 3))
                tmp_dict[k1] = tmp_lst
            print(tmp_dict)
            return
        except:
            pass


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
