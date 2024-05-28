#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

from helpers.datasets import partition_data
from helpers.synthesizers import AdvSynthesizer
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar2
from models.vgg import VGG9
from models.cnn import CNN_MNIST, CNN_CIFAR, CNN_EMNIST
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from models.vit import deit_tiny_patch16_224
from models.wrn import wrn_16_1, wrn_40_1
from models.efficientnet import effnetv2_s
from models.mobilenetv2 import MobileNetV2
from models.resnet import CustomResNet50

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)


# def init_model(idx, dataset):
#     if idx == 0 or idx == 1:
#         net = CustomResNet50(num_classes=47 if dataset == "emnist" else 10).to(args.device)
#     elif idx == 2 or idx == 3:
#         net = effnetv2_s(dataset=dataset).to(args.device)
#     elif idx == 4 or idx == 5:
#         net = MobileNetV2(dataset=dataset).to(args.device)  # cnn2
#     elif idx == 6 or idx == 7:
#         net = VGG9(dataset).to(args.device)
#     elif idx == 8 or idx == 9:
#         if dataset == "emnist":
#             net = CNN_EMNIST().to(args.device)
#         elif "mnist" in dataset:
#             net = CNN_MNIST(dataset).to(args.device)
#         else:
#             net = CNN_CIFAR().to(args.device)
#     return net


# def init_model(idx, dataset):
#     net = VGG9(dataset).to(args.device)
#     return net

def init_model(idx, dataset):
    if idx < 5:
        # net = resnet18(num_classes=10).cuda()
        net = VGG9(dataset).to(args.device)
    else:
        # net = CNNCifar2().cuda()
        if dataset == "emnist":
            net = CNN_EMNIST().to(args.device)
        elif "mnist" in dataset:
            net = CNN_MNIST(dataset).to(args.device)
        else:
            net = CNN_CIFAR().to(args.device)
    return net


# def init_model(idx, dataset):
#     if idx < 3:
#         # net = resnet18(num_classes=10).cuda()
#         net = VGG9(dataset).to(args.device)
#     else:
#         # net = CNNCifar2().cuda()
#         if dataset == "emnist":
#             net = CNN_EMNIST().to(args.device)
#         elif "mnist" in dataset:
#             net = CNN_MNIST(dataset).to(args.device)
#         else:
#             net = CNN_CIFAR().to(args.device)
#     return net


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, client_id):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.model = init_model(client_id, args.dataset)
        self.client_id = client_id

    def update_weights(self):
        self.model.train()
        # if self.client_id == 0:
        #     self.args.lr = 0.001
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
        #                             momentum=0.9)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        local_acc_list = []
        for iter in tqdm(range(self.args.local_ep)):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.model.zero_grad()
                # ---------------------------------------
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
                # ---------------------------------------
                loss.backward()
                optimizer.step()
            acc, test_loss = test(self.model, test_loader, args.device)
            local_acc_list.append(acc)

        return self.model.state_dict(), np.array(local_acc_list)


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    # parser.add_argument('--iid', type=int, default=1,
    #                     help='Default set to IID. Set to 0 for non-IID.')

    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for BN regularization')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--device', default="cuda:1", type=str,
                        help='seed for initializing training.')
    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        num_model = len(self.models)
        for i in range(num_model):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / num_model

        return logits_e


def kd_train(synthesizer, model, criterion, optimizer):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda()
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        pass
        # torch.save(state, filename)


def get_cls_num_list(traindata_cls_counts, dataset):
    cls_num_list = []
    for key, val in traindata_cls_counts.items():
        temp = [0] * 47 if dataset == "emnist" else [0] * 10
        for key_1, val_1 in val.items():
            temp[key_1] = val_1
        cls_num_list.append(temp)

    return cls_num_list


if __name__ == '__main__':

    args = args_parser()
    setup_seed(args.seed)
    # load dataset and user groups
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL
    global_model = VGG9(args.dataset).cuda()
    # global_model = effnetv2_s(args.dataset).cuda()
    # global_model = resnet18(num_classes=10).cuda()
    # global_model = wrn_40_1(num_classes=10, dropout_rate=0).cuda()
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    print("type(traindata_cls_counts)", type(traindata_cls_counts))
    cls_num_list = get_cls_num_list(traindata_cls_counts, args.dataset)
    print(cls_num_list)
    acc_list = []
    if args.type == "pretrain":
        test_lists = []
        # ===============================================
        idxs_users = range(args.num_users)
        for idx in idxs_users:
            print("client {}".format(idx))
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], client_id=idx)
            w, np_val = local_model.update_weights()
            local_weights.append(copy.deepcopy(w))
            test_lists.append(np_val)
        torch.save(local_weights, 'heter_{}.pkl'.format(args.beta))
        np.save("heter_acc_beta{}.npy".format(args.beta), np.array(test_lists))
        # local_weights = torch.load('heter.pkl')
        model_list = []
        for i in range(len(local_weights)):
            net = init_model(i, dataset=args.dataset)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
            test(net, test_loader, args.device)
        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader, args.device)
        # ===============================================
    else:
        # ===============================================
        local_weights = torch.load('heter_{}.pkl'.format(args.beta))
        model_list = []
        for i in range(len(local_weights)):
            net = init_model(i, args.dataset)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
            test(net, test_loader, args.device)
        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader, args.device)
        # data generator
        nz = args.nz
        nc = 3 if args.dataset in ["cifar10", "svhn"] else 1
        img_size = 32 if args.dataset in ["cifar10", "svhn"] else 28
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        args.cur_ep = 0
        img_size2 = (3, 32, 32) if args.dataset in ["cifar10", "svhn"] else (1, 28, 28)
        synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                     nz=nz, num_classes=47 if args.dataset == "emnist" else 10, img_size=img_size2,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size,
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir=args.save_dir, dataset=args.dataset, device=args.device)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        criterion = KLDiv(T=args.T)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []
        for epoch in tqdm(range(args.epochs)):
            # 1. Data synthesis
            synthesizer.gen_data(args.cur_ep, device=args.device)  # g_steps
            args.cur_ep += 1
            kd_train(synthesizer, [global_model, ensemble_model], criterion, optimizer)  # # kd_steps
            acc, test_loss = test(global_model, test_loader, device=args.device)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)
            _best_ckpt = 'df_ckpt/{}.pth'.format(args.other)
            print("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)
        np.save("distill_acc_{}_beta{}.npy".format(args.dataset, args.beta), np.array(distill_acc))

        # ===============================================
