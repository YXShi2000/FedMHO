# -*- coding: utf-8 -*-
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pcode.local_training.compressor as compressor
import pcode.local_training.random_reinit as random_reinit
import pcode.datasets.mixup_data as mixup
import pcode.create_model as create_model
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.create_metrics as create_metrics
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat, display_test_stat
from pcode.utils.timer import Timer
from pcode.utils.stat_tracker import RuntimeTracker
from skimage.feature import hog
from copy import deepcopy


class Worker(object):
    def __init__(self, conf):
        self.conf = conf
        self.conf.n_participated = int(conf.n_clients * conf.participation_ratio)
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")
        self.ml_model_lst = ["svm", "lr", "lr_s"]
        if conf.dynamic_local_n_epochs:
            self.delta_epoch = int(conf.local_n_epochs / 8)
            self.epoch_lst = []

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
            log_fn=conf.logger.log_metric,
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.dataset = create_dataset.define_dataset(self.conf, data=self.conf.data)
        # if conf.data == "mnist":  # image size 28x28 -> 32 x 32
        #     # print(type(self.dataset["train"].data))  # torchvision.datasets.mnist.MNIST
        #     pad = torch.nn.ZeroPad2d(padding=(2, 2, 2, 2))
        #     self.dataset["train"].data.data = pad(self.dataset["train"].data.data)
        conf.logger.log(f"Worker-{self.conf.graph.worker_id} defined dataset.")

        if conf.use_hog_feature:
            self.hog_train_dataset = self._get_hog_dataset(self.dataset["train"], is_train=True)
            # self.hog_test_dataset = self._get_hog_dataset(self.dataset["test"], is_train=False)
        conf.logger.log(f"Worker-{self.conf.graph.worker_id} defined hog dataset.")

        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )

        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # define the model compression operators.
        if conf.local_model_compression is not None:
            if conf.local_model_compression == "quantization":
                self.model_compression_fn = compressor.ModelQuantization(conf)

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self._recv_model_from_master()
            self._train()
            if self.conf.dynamic_local_n_epochs:
                print(self.arch, self.epoch_lst)
            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return

    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg_size = 6 if self.conf.dynamic_local_n_epochs else 3
        msg = torch.zeros((msg_size, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        if self.conf.dynamic_local_n_epochs:
            self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs, self.resnet_acc, self.lr_acc, self.svm_acc = (
                msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
            )
        else:
            self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
                msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
            )
        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        if self.arch in self.ml_model_lst:
            self.n_local_epochs = self.conf.ml_local_n_epochs
        if self.conf.dynamic_local_n_epochs:
            if self.conf.graph.comm_round == 1:
                self.pre_n_local_epochs = self.n_local_epochs
                self.epoch_lst.append(self.n_local_epochs)
                self.pre_acc = 0
            else:
                # get temperature round accuracy.
                if "resnet" in self.arch:
                    self.tmp_acc = self.resnet_acc
                elif "lr" in self.arch:
                    self.tmp_acc = self.lr_acc
                elif "svm" in self.arch:
                    self.tmp_acc = self.svm_acc
                else:
                    print("no such arch!")
                # chang epoch.
                if self.conf.graph.comm_round == 2:
                    self.alpha = 1
                else:
                    if self.pre_delta_acc < 0.01:
                        self.alpha = 0
                    else:
                        self.alpha = (self.tmp_acc - self.pre_acc) / (self.pre_delta_acc)
                    # regulate alpha.
                    if self.alpha > 1:
                        self.alpha = 1
                    if self.alpha < -1:
                        self.alpha = -1
                print("-------------------")
                print(self.arch, self.alpha)
                print("-------------------")
                # get dynamic local epochs
                self.n_local_epochs = self.pre_n_local_epochs - int(self.alpha * self.delta_epoch)
                self.epoch_lst.append(self.n_local_epochs)
                # update pre-based variable
                self.pre_delta_acc = math.fabs(self.tmp_acc - self.pre_acc)
                self.pre_acc = self.tmp_acc
                self.pre_n_local_epochs = self.n_local_epochs
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")

        self._update_data_partitioner()
        self._update_criterion()
        self.conf.logger.log(f"Worker-{self.conf.graph.worker_id} listened to master.")
        dist.barrier()

    def _update_data_partitioner(self):
        if not self.conf.use_hog_feature or self.arch not in self.ml_model_lst:
            self.data_partitioner.data = self.dataset["train"]
        else:
            self.data_partitioner.data = self.hog_train_dataset

    def _update_criterion(self):
        if "svm" in self.arch:
            self.criterion = nn.MultiMarginLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        random_reinit.random_reinit_model(self.conf, self.model)
        # use for self-distillation
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def _train(self):
        self.conf.logger.log(f"Worker-{self.conf.graph.worker_id} training on {self.device}.")
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)
            # self.model = torch.nn.DataParallel(self.model)

        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=None,
            # client_id starts from 1 to the # of clients.
            localdata_id=self.conf.graph.client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner)
        # print(type(self.train_loader.dataset.data))
        # print(type(self.train_loader.dataset.data.data))
        # print(self.arch, self.train_loader.dataset.data.data.data.shape)
        # define optimizer, scheduler and runtime tracker.

        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer,
            use_lr_ml=self.arch in self.ml_model_lst, arch=self.arch,
        )
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        # self.test_tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )

        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            for _input, _target in self.train_loader:
                # load data
                data_batch = create_dataset.load_data_batch(
                    self.conf, _input, _target, is_training=True
                )
                # with self.timer("load_data", epoch=self.scheduler.epoch_):
                #     print(self.arch, "loading data...")
                #     print(_input, _target)
                #     data_batch = create_dataset.load_data_batch(
                #         self.conf, _input, _target, is_training=True
                #     )

                # inference and get current performance.
                self.optimizer.zero_grad()
                loss, output = self._inference(data_batch)

                # in case we need self distillation to penalize the local training
                # (avoid catastrophic forgetting).
                self._local_training_with_self_distillation(
                    loss, output, data_batch
                )
                # with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                #     self.optimizer.zero_grad()
                #     loss, output = self._inference(data_batch)
                #
                #     # in case we need self distillation to penalize the local training
                #     # (avoid catastrophic forgetting).
                #     self._local_training_with_self_distillation(
                #         loss, output, data_batch
                #     )

                loss.backward()
                if "resnet" in self.arch:
                    self._add_grad_from_prox_regularized_loss()
                self.optimizer.step()
                self.scheduler.step()
                # with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                #     loss.backward()
                #     if "resnet" in self.arch:
                #         self._add_grad_from_prox_regularized_loss()
                #     self.optimizer.step()
                #     self.scheduler.step()

                # efficient local training.
                # with self.timer("compress_model", epoch=self.scheduler.epoch_):
                #     if hasattr(self, "model_compression_fn"):
                #         self.model_compression_fn.compress_model(
                #             param_groups=self.optimizer.param_groups
                #         )

                # display the logging info.
                # display_training_stat(self.conf, self.scheduler, self.tracker)

                # display tracking time.
                # if (
                #         self.conf.display_tracked_time
                #         and self.scheduler.local_index % self.conf.summary_freq == 0
                # ):
                #     self.conf.logger.log(self.timer.summary())

                # check divergence.
                if self.tracker.stat["loss"].avg > 1e3 or np.isnan(
                        self.tracker.stat["loss"].avg
                ):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round():
                    self._terminate_comm_round()
                    return

            # display the logging info.
            # test model on testset
            # for _input, _target in self.test_loader:
            #     data_batch = create_dataset.load_data_batch(
            #         self.conf, _input, _target, is_training=False
            #     )
            #     self._get_test_perf(data_batch)
            display_training_stat(self.conf, self.scheduler, self.tracker)
            # print("----------arch:",self.arch,"--------test---------")
            # display_test_stat(self.conf, self.scheduler, self.test_tracker)

            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            # self.test_tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _get_test_perf(self, data_batch):
        output = self.model(data_batch["input"])
        loss = self.criterion(output, data_batch["target"])
        performance = self.metrics.evaluate(loss, output, data_batch["target"])
        if self.test_tracker is not None:
            self.test_tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        output = self.model(data_batch["input"])

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            # add regularization i.e. Full loss = data loss + regularization loss
            if "svm" in self.arch:
                weight = self.model.linear.weight.squeeze()
                if self.conf.svm_l_type == 'L1':  # add L1 (LASSO) loss
                    # print("-----------------")
                    # print(loss)
                    loss += self.conf.svm_c * torch.sum(torch.abs(weight))
                    # print(loss)
                    # print("-----------------")
                    # loss += torch.mean(torch.sum(torch.abs(weight)))
                elif self.conf.svm_l_type == 'L2':  # add L2 (Ridge) loss
                    # loss += torch.sum(weight * weight)
                    loss += self.conf.svm_c * torch.mean(torch.sum(weight * weight))
                elif self.conf.svm_l_type == 'L1L2':  # add Elastic net (beta*L2 + L1) loss
                    # loss += torch.sum(self.conf.svm_beta * weight * weight + torch.abs(weight))
                    loss += self.conf.svm_c * torch.mean(
                        torch.sum(self.conf.svm_beta * weight * weight + torch.abs(weight)))

            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

    def _add_grad_from_prox_regularized_loss(self):
        assert self.conf.local_prox_term >= 0
        if self.conf.local_prox_term != 0:
            assert self.conf.weight_decay == 0
            assert self.conf.optimizer == "sgd"
            assert self.conf.momentum_factor == 0

            for _param, _init_param in zip(
                    self.model.parameters(), self.init_model.parameters()
            ):
                if _param.grad is not None:
                    _param.grad.data.add_(
                        (_param.data - _init_param.data) * self.conf.local_prox_term
                    )

    def _local_training_with_self_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0:
            loss = loss * (
                    1 - self.conf.self_distillation
            ) + self.conf.self_distillation * self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=self.init_model(data_batch["input"])
                               / self.conf.self_distillation_temperature,
            )
        return loss

    def _divergence(self, student_logits, teacher_logits):
        # print("student_logits:", student_logits)
        # print("teacher_logits:", teacher_logits)
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier()

    def _terminate_comm_round(self):
        self.model = self.model.cpu()
        del self.init_model
        self.scheduler.clean()
        self.conf.logger.save_json()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False
        # return True if self.conf.epoch_ >= self.n_local_epochs else False

    def _get_hog_dataset(self, original_dataset, is_train):
        hog_dataset = deepcopy(original_dataset)
        # if is_train: type(hog_dataset.data)=torchvision.datasets.cifar.CIFAR10
        # else type(hog_dataset)=torchvision.datasets.cifar.CIFAR10
        tmp_dataset = hog_dataset.data if is_train else hog_dataset
        print("worker||original data shape:", tmp_dataset.data.shape,
              "|| original targets shape:", len(tmp_dataset.targets))
        tmp_dataset_data = []
        for _, image in enumerate(tmp_dataset.data):
            if self.conf.data != "mnist":
                image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            hog_feature = hog(image, orientations=8,
                              pixels_per_cell=((8, 8) if self.conf.data != "mnist" else (7, 7)),
                              cells_per_block=(2, 2))
            tmp_dataset_data.append(hog_feature)
        if is_train:
            hog_dataset.data.data = np.array(tmp_dataset_data)
        else:
            hog_dataset.data = np.array(tmp_dataset_data)
        print("worker||hog data shape:", tmp_dataset.data.shape,
              "|| hog targets shape:", len(tmp_dataset.targets))
        return hog_dataset
