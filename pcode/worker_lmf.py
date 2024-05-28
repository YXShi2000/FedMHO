# -*- coding: utf-8 -*-
import copy
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
from pcode.utils.logging import display_training_stat
from pcode.utils.timer import Timer
from pcode.utils.stat_tracker import RuntimeTracker
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CVAELoss(nn.Module):
    def __init__(self):
        super(CVAELoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class Worker(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
            log_fn=conf.logger.log_metric,
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data)
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

        # define the model compression operators.
        if conf.local_model_compression is not None:
            if conf.local_model_compression == "quantization":
                self.model_compression_fn = compressor.ModelQuantization(conf)

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )

    def run(self):
        while True:
            self._listen_to_master()

            self._define_criterion()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self._recv_model_from_master()
            self._train()
            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return

    def _define_criterion(self):
        if "vae" in self.arch:
            # self.criterion1 = nn.BCELoss(reduction='sum')
            self.criterion1 = nn.BCEWithLogitsLoss(reduction='sum')
            self.criterion2 = lambda mu, sigma: -0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu.pow(2) - sigma ** 2)
            self.cifar_criterion = CVAELoss()

        else:
            self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        random_reinit.random_reinit_model(self.conf, self.model)
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def _train(self):
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.cuda()

        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=self.conf.graph.client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )

        if self.conf.data == "cifar10" and "vae" in self.arch:
            print("---------- before aug ----------\n")
            aug_transform = [transforms.RandomRotation(10),
                             transforms.RandomResizedCrop(size=32, scale=(0.9, 0.9)),
                             transforms.RandomAffine(degrees=0, translate=(0.2, 0.1)),
                             transforms.RandomAffine(degrees=0, translate=(0.1, 0.2))]
            # original_data_loader = DataLoader(self.dataset["train"], batch_size=64, shuffle=True)
            for original_batch, original_label in self.train_loader:
                augmented_images, augmented_targets = copy.deepcopy(original_batch), copy.deepcopy(original_label)
                break
            print(augmented_images.shape, augmented_targets.shape)
            for idx, (original_batch, original_label) in enumerate(self.train_loader):
                if idx > 0:
                    augmented_images = torch.cat([augmented_images, original_batch])
                    augmented_targets = torch.cat([augmented_targets, original_label])

                for trans in aug_transform:
                    augmented_images = torch.cat([augmented_images, trans(original_batch)])
                    augmented_targets = torch.cat([augmented_targets, original_label])

            augmented_dataset = torch.utils.data.TensorDataset(augmented_images, augmented_targets)
            self.train_loader = DataLoader(augmented_dataset, batch_size=self.conf.batch_size, shuffle=True)
            print("---------- finish aug ----------\n", augmented_images.shape)

        # define optimizer, scheduler and runtime tracker.
        if "vae" not in self.arch:
            if "cnn" in self.arch:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9)
            else:
                self.optimizer = create_optimizer.define_optimizer(
                    self.conf, model=self.model, optimizer_name=self.conf.optimizer
                )
        else:
            if self.conf.data in ["svhn", "cifar10"]:
                lr = 1e-3
            elif self.conf.data == "emnist":
                lr = 1e-3 if self.model.name == "cvae_large" else 3e-3
            else:
                lr = 1e-3 if self.model.name == "cvae_large" else 5e-2
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )

        # entering local updates and will finish only after reaching the expected local_n_epochs.
        if "vae" in self.arch and self.conf.data == "emnist":
            for epoch in range(1, 51):
                self.model.train()
                train_loss = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    self.optimizer.zero_grad()

                    output, mu, logvar = self.model(data, target)
                    loss = self.loss_function_emnist(output, data, mu, logvar)

                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()

                    if batch_idx % 100 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                   100. * batch_idx / len(self.train_loader), loss.item() / len(data)))

                print("[ client", self.conf.graph.client_id, ']  ====> Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                                                                              train_loss / len(
                                                                                                                  self.train_loader.dataset)))
            self._terminate_comm_round()
            return

        else:
            while True:
                self.loss = 0
                for _input, _target in self.train_loader:
                    # load data
                    with self.timer("load_data", epoch=self.scheduler.epoch_):
                        data_batch = create_dataset.load_data_batch(
                            self.conf, _input, _target, is_training=True
                        )

                    # inference and get current performance.
                    with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                        self.optimizer.zero_grad()
                        loss, _ = self._inference(data_batch)

                        # in case we need self distillation to penalize the local training
                        # (avoid catastrophic forgetting).
                        # self._local_training_with_self_distillation(
                        #     loss, output, data_batch
                        # )

                    with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                        loss.backward()
                        # self._add_grad_from_prox_regularized_loss()
                        self.optimizer.step()
                        self.scheduler.step()

                    # efficient local training.
                    with self.timer("compress_model", epoch=self.scheduler.epoch_):
                        if hasattr(self, "model_compression_fn"):
                            self.model_compression_fn.compress_model(
                                param_groups=self.optimizer.param_groups
                            )

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
                if "vae" not in self.arch:
                    display_training_stat(self.conf, self.scheduler, self.tracker)

                # refresh the logging cache at the end of each epoch.
                self.tracker.reset()
                if self.conf.logger.meet_cache_limit():
                    self.conf.logger.save_json()

                if "vae" in self.arch:
                    print("[ client", self.conf.graph.client_id, "]   epoch:", int(self.conf.epoch_), "training loss:",
                          self.loss)

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        if self.arch == "vae":
            re_imgs, mu, sigma = self.model(data_batch["input"])
            output = None
        elif "cvae" in self.arch:
            if self.model.name == "cvae_large":
                recon_x, mean, log_var = self.model(data_batch["input"], data_batch["target"])
            elif self.conf.data in ["mnist", "fashionmnist"]:
                recon_x, mean, log_var, z = self.model(data_batch["input"], data_batch["target"])
            else:
                recon_x, mean, log_var = self.model(data_batch["input"], data_batch["target"])
            output = None
        else:
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
            if self.arch == "vae":
                alpha = 0.6
                loss_re = self.criterion1(re_imgs, data_batch["input"].view(data_batch["input"].size(0), -1))
                loss_norm = self.criterion2(mu, sigma)
                loss = alpha * loss_re + (1 - alpha) * loss_norm
                performance = None
            elif "cvae" in self.arch:
                if self.conf.data == "emnist":
                    loss = self.loss_function_emnist(recon_x, data_batch["input"], mean, log_var)
                elif "mnist" in self.conf.data:
                    loss = self.loss_fn_cvae(recon_x, data_batch["input"], mean, log_var)
                else:
                    loss = self.cifar_criterion(recon_x, data_batch["input"], mean, log_var)
                self.loss += loss.item()
                # print(loss)
                performance = None
            else:
                loss = self.criterion(output, data_batch["target"])
                performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if "vae" in self.arch:
            pass
        else:
            if self.tracker is not None:
                self.tracker.update_metrics(
                    [loss.item()] + performance, n_samples=data_batch["input"].size(0)
                )
        return loss, output

    def loss_fn_cvae(self, recon_x, x, mean, log_var):
        # print("recon_x", recon_x, "\nx:", x)
        if "mnist" in self.conf.data:
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
        else:
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, 3 * 32 * 32), x.view(-1, 3 * 32 * 32), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # print("[client", self.conf.graph.client_id, "] BCE loss:", BCE.item(), "KL loss:", KLD.item())

        return (BCE + KLD) / x.size(0)

    def loss_function_emnist(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

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
        if self.conf.one_shot:
            return True if self.conf.epoch_ >= self.conf.local_n_epochs else False
        if "vae" in self.arch:
            if self.conf.cvae_local_epoch > 0:
                threshold = self.conf.cvae_local_epoch
            elif self.conf.data == "mnist":
                threshold = 30
            elif self.conf.data == "emnist":
                threshold = 50
            elif self.conf.data == "fashionmnist":
                threshold = 40
            elif self.conf.data == "svhn":
                threshold = 40
            else:
                threshold = 30
            return True if self.conf.epoch_ >= threshold else False
        else:
            return True if self.conf.epoch_ >= self.conf.local_n_epochs else False
