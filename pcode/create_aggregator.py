# -*- coding: utf-8 -*-
import json
import copy
import torch
import numpy as np

import pcode.datasets.prepare_data as prepare_data
import pcode.datasets.partition_data as partition_data
import pcode.aggregation.fedavg as fedavg
import pcode.aggregation.noise_knowledge_transfer as noise_knowledge_transfer
import pcode.aggregation.semi_supervised as semi_supervised
import pcode.aggregation.federated_average_ensemble as federated_average_ensemble
from skimage.feature import hog


class Aggregator(object):
    def __init__(
            self, conf, criterion, metrics, dataset, test_loaders, clientid2arch
    ):
        self.conf = conf
        self.logger = conf.logger
        self.criterion = criterion
        self.metrics = metrics
        self.dataset = dataset
        self.test_loaders = test_loaders
        self.clientid2arch = clientid2arch

        # for the ease of aggregation.
        self.data_info = self._define_aggregation_data(return_loader=True)
        if conf.use_hog_feature and "federated_average" not in conf.fl_aggregate["scheme"]:
            self.hog_data_info = self._define_aggregation_data(return_loader=True, get_hog_feature=True)

        # define the aggregation function.
        self._define_aggregate_fn()

    def _define_aggregate_fn(self):
        if (
                self.conf.fl_aggregate is None
                or self.conf.fl_aggregate["scheme"] == "federated_average"
        ):
            self.aggregate_fn = None
        elif self.conf.fl_aggregate["scheme"] == "federated_average_ensemble":
            self.aggregate_fn = self._s6_federated_average_ensemble()
        elif self.conf.fl_aggregate["scheme"] == "noise_knowledge_transfer":
            self.aggregate_fn = self._s7_noise_knowledge_transfer()
        elif self.conf.fl_aggregate["scheme"] == "semi_supervised":
            self.aggregate_fn = self._s11_semi_supervised()
        else:
            raise NotImplementedError

    def _s6_federated_average_ensemble(self):
        def f(**kwargs):
            _client_models = federated_average_ensemble.aggregate(
                conf=self.conf,
                fedavg_models=kwargs["fedavg_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                test_data_loader=self.test_loaders[0],
            )
            return _client_models

        return f

    def _s11_semi_supervised(self):
        def f(**kwargs):
            _client_models = semi_supervised.aggregate(
                conf=self.conf,
                fedavg_models=kwargs["fedavg_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                test_data_loader=self.test_loaders[0],
                total_semi_supervised_data=kwargs["total_semi_supervised_data"],
                train_global_model=kwargs["train_global_model"],
            )
            return _client_models

        return f

    def _s1_federated_average(self):
        # global-wise averaging scheme.
        def f(**kwargs):
            return fedavg.fedavg(
                conf=self.conf,
                clientid2arch=self.clientid2arch,
                n_selected_clients=kwargs["n_selected_clients"],
                flatten_local_models=kwargs["flatten_local_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                # val_data_loader=self.data_info["self_val_data_loader"],
                test_data_loader=self.test_loaders[0]
            )

        return f

    def _s7_noise_knowledge_transfer(self):
        def f(**kwargs):
            _client_models = noise_knowledge_transfer.aggregate(
                conf=self.conf,
                fedavg_models=kwargs["fedavg_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                fa_val_perf=kwargs["performance"],
                distillation_sampler=self.data_info["sampler"],
                hog_distillation_sampler=None if not self.conf.use_hog_feature else self.hog_data_info["sampler"],
                distillation_data_loader=self.data_info["data_loader"],
                hog_distillation_data_loader=None if not self.conf.use_hog_feature else self.hog_data_info[
                    "data_loader"],
                val_data_loader=self.data_info["self_val_data_loader"],
                hog_val_data_loader=None if not self.conf.use_hog_feature else self.hog_data_info[
                    "self_val_data_loader"],
                test_data_loader=self.test_loaders[0],
                logits_remain=kwargs["logits_remain"],
            )
            return _client_models

        return f

    def aggregate(
            self,
            master_model,
            client_models,
            flatten_local_models,
            aggregate_fn_name=None,
            kd_train_loader=None,
            kd_test_loader=None,
            save_model_name=None,
            **kwargs,
    ):
        n_selected_clients = len(flatten_local_models)

        # apply advanced aggregate_fn.
        self.logger.log(
            f"Aggregator via {aggregate_fn_name if aggregate_fn_name is not None else self.conf.fl_aggregate['scheme']}: {f'scheme={json.dumps(self.conf.fl_aggregate)}' if self.conf.fl_aggregate is not None else ''}"
        )
        _aggregate_fn = (
            self.aggregate_fn
            if aggregate_fn_name is None
            else getattr(self, aggregate_fn_name)()
        )
        return _aggregate_fn(
            master_model=master_model,
            client_models=client_models,
            flatten_local_models=flatten_local_models,
            n_selected_clients=n_selected_clients,
            kd_train_loader=kd_train_loader,
            kd_test_loader=kd_test_loader,
            save_model_name=save_model_name,
            **kwargs,
        )

    def _define_aggregation_data(self, return_loader=True, get_hog_feature=False):
        # init.
        fl_aggregate = self.conf.fl_aggregate

        # prepare the data.
        if self.dataset["val"] is not None:
            if get_hog_feature:
                tmp_dataset_data = []
                tmp_val_dataset = copy.deepcopy(self.dataset["val"])
                # print(type(tmp_val_dataset.data))  # # torchvision.datasets.cifar.CIFAR10
                print("proxy val data converting to hog features...")
                for idx, image in enumerate(tmp_val_dataset.data.data):
                    image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
                    hog_feature = hog(image, orientations=8,
                                      pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2))
                    tmp_dataset_data.append(hog_feature)
                tmp_val_dataset.data.data = np.array(tmp_dataset_data)
            # prepare the dataloader.
            data_loader = torch.utils.data.DataLoader(
                tmp_val_dataset if get_hog_feature else self.dataset["val"],
                batch_size=self.conf.batch_size,
                shuffle=False,
                num_workers=self.conf.num_workers,
                pin_memory=self.conf.pin_memory,
                drop_last=False,
            )
            # define things to return.
            things_to_return = {"self_val_data_loader": data_loader}
        else:
            things_to_return = {}

        if "data_source" in fl_aggregate and "other" in fl_aggregate["data_source"]:
            assert (
                    "data_name" in fl_aggregate
                    and "data_scheme" in fl_aggregate
                    and "data_type" in fl_aggregate
            )

            # create dataset.
            self.logger.log(f'create data={fl_aggregate["data_name"]} for aggregation.')
            dataset = prepare_data.get_dataset(
                self.conf,
                fl_aggregate["data_name"],
                datasets_path=self.conf.data_dir
                if "data_dir" not in fl_aggregate
                else fl_aggregate["data_dir"],
                split="train",
            )
            # print(type(dataset))  # torchvision.datasets.cifar.CIFAR100
            if get_hog_feature:
                tmp_dataset_data = []
                print("proxy data converting to hog features...")
                for idx, image in enumerate(dataset.data):
                    image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
                    hog_feature = hog(image, orientations=8, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2))
                    tmp_dataset_data.append(hog_feature)
                dataset.data = np.array(tmp_dataset_data)

            self.logger.log(
                f'created data={fl_aggregate["data_name"]} for aggregation with size {len(dataset)}.'
            )

            # sample the indices from the dataset.
            if fl_aggregate["data_scheme"] == "random_sampling":
                assert "data_percentage" in fl_aggregate
                sampler = partition_data.DataSampler(
                    self.conf,
                    data=dataset,
                    data_scheme=fl_aggregate["data_scheme"],
                    data_percentage=fl_aggregate["data_percentage"],
                )
            elif fl_aggregate["data_scheme"] == "class_selection":
                assert "num_overlap_class" in fl_aggregate
                assert "num_total_class" in fl_aggregate
                assert self.conf.data == "cifar100"
                assert "imagenet" in self.conf.fl_aggregate["data_name"]

                selected_imagenet_classes = partition_data.get_imagenet1k_classes(
                    num_overlap_classes=int(fl_aggregate["num_overlap_class"]),
                    random_state=self.conf.random_state,
                    num_total_classes=int(
                        fl_aggregate["num_total_class"]
                    ),  # for cifar-100
                )
                sampler = partition_data.DataSampler(
                    self.conf,
                    data=dataset,
                    data_scheme=fl_aggregate["data_scheme"],
                    data_percentage=fl_aggregate["data_percentage"]
                    if "data_percentage" in fl_aggregate
                    else None,
                    selected_classes=selected_imagenet_classes,
                )
            else:
                raise NotImplementedError("invalid data_scheme")

            sampler.sample_indices()

            # define things to return.
            things_to_return.update({"sampler": sampler})

            if return_loader:
                data_loader = torch.utils.data.DataLoader(
                    sampler.use_indices(),
                    batch_size=self.conf.batch_size,
                    shuffle=fl_aggregate["randomness"]
                    if "randomness" in fl_aggregate
                    else True,
                    num_workers=self.conf.num_workers,
                    pin_memory=self.conf.pin_memory,
                    drop_last=False,
                )
            things_to_return.update({"data_loader": data_loader})
        return things_to_return
