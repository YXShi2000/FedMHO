# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import pcode.aggregation.utils as agg_utils
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset
from torch.utils import data
from copy import deepcopy
import numpy as np
# import matplotlib.pyplot as plt
import pcode.master_utils as master_utils
import pcode.models as models
import os
from pcode.utils.data_clean import data_clean
from pcode.models.vgg import VGG9
import torch.nn.functional as F
from pcode.datasets.partition_data import get_shared_targets_of_partitions
import matplotlib.pyplot as plt



class Mydatasetpro(data.Dataset):
    def __init__(self, img_list, labels, transform):
        self.imgs = img_list
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        # print(type(img), img.shape)
        label = self.labels[index]
        img = Image.fromarray(img)
        # print(img.size)
        data = self.transforms(img)
        # print(data.shape)
        return data, label

    def __len__(self):
        return len(self.imgs)


def get_perf(conf, model, criterion, metrics, test_data_loader):
    perf = master_utils.validate(
        conf,
        coordinator=None,
        model=model,
        criterion=criterion,
        metrics=metrics,
        data_loader=test_data_loader,
        label=None,
        display=False,
    )
    print("----------------\n", perf, "\n----------------")


def get_VAE_list(num_class, local_models):
    VAE_lst, VAE_data_partition_lst = [], []
    shared_targets_of_partitions = list(get_shared_targets_of_partitions())
    # print(shared_targets_of_partitions)
    for idx, local_model in enumerate(list(local_models.values())):
        if local_model.name in ["vae", "VAE", "cvae", "cvae_mnist", "cvae_cifar", "cvae_large"]:
            VAE_lst.append(local_model)
            tmp = shared_targets_of_partitions[idx][1]
            result = [0] * num_class
            for pair in tmp:
                index, value = pair
                result[index] = value

            VAE_data_partition_lst.append(result)
    print("get %d vae model" % len(VAE_lst))
    print(VAE_data_partition_lst)
    return VAE_lst, VAE_data_partition_lst


def get_resnet_vae_model(conf, fedavg_models):
    # arch = determine_arch(conf, client_id=1, use_complex_arch=True)
    if "resnet" in conf.arch:
        resnet_model_raw = models.__dict__["resnet"](conf, arch=conf.arch)
    elif "vgg" in conf.arch:
        resnet_model_raw = models.__dict__["vgg"](conf)
    elif "cnn" in conf.arch:
        resnet_model_raw = models.__dict__["cnn"](conf, arch="cnn")
    elif "efficientnet" in conf.arch:
        resnet_model_raw = models.__dict__["efficientnet"](conf)
    cvae_model_raw = models.__dict__["cvae"](conf, arch="cvae")

    if not conf.one_shot:
        for arch, fedavg_model in fedavg_models.items():
            if arch == conf.arch:
                resnet_model_raw.load_state_dict(fedavg_model.state_dict())
            elif "cave" in arch:
                cvae_model_raw.load_state_dict(fedavg_model.state_dict())
    # get_perf(conf, resnet_model_raw, criterion, metrics, test_data_loader)
    return resnet_model_raw, cvae_model_raw


def get_synthesis_data_list(conf, synthesis_sample, latent_size, device, **args):
    synthesis_data_list, synthesis_data_label_list = [], []
    num_class = 47 if conf.data == "emnist" else 10
    sample_per_model = int(synthesis_sample / len(args["VAE_lst"]))
    sample_per_class = int(synthesis_sample / num_class)
    sample_per_class_per_model = int(sample_per_class / len(args["VAE_lst"]))
    VAE_data_partition_lst = args["VAE_data_partition_lst"]

    # from local vae (cvae) models
    # for m in args["VAE_lst"]:
    #     for i in range(sample_per_model):
    #         sample = torch.randn(1, latent_size).to(device)
    #         generate_img = m.decoder(sample)[0].view(28, 28)
    #         # print(generate_img.shape)  # torch.Size([28, 28])
    #         plt.matshow(generate_img.cpu().detach().numpy())
    #         plt.show()
    #         plt.savefig("%d.png" % i)
    #         synthesis_data_list.append(generate_img.cpu().detach().numpy())


    # for idx, model in enumerate(args["VAE_lst"]):
    #     print("generating model...", idx)
    #     model.eval()
    #     # Set up the figure with appropriate spacing
    #     fig, axs = plt.subplots(num_class, 30, figsize=(30, num_class))
    #     fig.subplots_adjust(hspace=0.2, wspace=0.2)
    #
    #     for i in range(num_class):
    #         for j in range(30):
    #             z = torch.randn(1, latent_size).to(device)
    #             y = torch.zeros(1, num_class).to(device)
    #             y[0, i] = 1  # one-hot encoding
    #
    #             with torch.no_grad():
    #                 output = model.decoder(torch.cat((z, y), dim=1))
    #             generated_image = output.view(28, 28).cpu().numpy()
    #
    #             axs[i, j].imshow(generated_image, cmap='gray')
    #             axs[i, j].axis('off')
    #     # Save and show the figure
    #     plt.savefig(f'model_{idx}_generated_images.png')
    #     # plt.show()


    one_shot_data_list = []
    for p in range(num_class):
        element_list = [l[p] for l in VAE_data_partition_lst]
        weights = [(element ** 2) for element in element_list]
        total_weight = sum(weights)
        if total_weight == 0:
            normalized_weights = [1 / len(weights) for weight in weights]
        else:
            normalized_weights = [weight / total_weight for weight in weights]
        # print("class", p, normalized_weights)

        synthesis_data_per_class_list, synthesis_data_label_per_class_list = [], []
        for idx, m in enumerate(args["VAE_lst"]):
            sample_per_class_per_model = int(normalized_weights[idx] * sample_per_class)
            if sample_per_class_per_model == 0:
                continue
            if "mnist" in args["dataset"]:
                c = torch.full((sample_per_class_per_model, 1), p, dtype=torch.long, device=device)
            else:
                c = torch.full((sample_per_class_per_model,), p, dtype=torch.long).to(device)
            # mnist, fashion-mnist, emnist
            if "mnist" in args["dataset"]:
                if m.name == "cvae_large" and args["dataset"] != "emnist":
                    z = torch.randn(sample_per_class_per_model, 20).to(device)
                    labels_onehot = F.one_hot(c, 10).float().view(sample_per_class_per_model, 10)
                    z_combined = torch.cat((z, labels_onehot), dim=1)
                    # print(z_combined.shape)
                    x = m.decoder(z_combined)
                elif args["dataset"] == "emnist":
                    z = torch.randn([c.size(0), latent_size]).to(device)
                    y = torch.zeros(c.size(0), num_class).to(device)
                    y[:, p] = 1
                    x = m.decoder(torch.cat((z, y), dim=1))
                else:
                    z = torch.randn([c.size(0), latent_size]).to(device)
                    x = m.inference(z, c=c)

                for i in range(sample_per_class_per_model):
                    generate_img = x[i].view(28, 28).cpu().data.numpy()
                    synthesis_data_per_class_list.append(generate_img)
                    synthesis_data_label_per_class_list.append(p)
            # svhn, cifar-10
            else:
                if m.name == "cvae_large":
                    z = torch.randn(sample_per_class_per_model, 20).to(device)
                    labels_onehot = F.one_hot(c, num_class).float().view(sample_per_class_per_model, 10)
                    z_combined = torch.cat((z, labels_onehot), dim=1)
                    # print(z_combined.shape)
                    x = m.decoder(z_combined)
                else:
                    z = torch.randn([sample_per_class_per_model, 20]).to(device)
                    y_onehot = torch.zeros(sample_per_class_per_model, 10).to(device)
                    y_onehot.scatter_(1, c.unsqueeze(1), 1)
                    z_y = torch.cat([z, y_onehot], dim=1)
                    x = m.decoder(z_y)
                for i in range(sample_per_class_per_model):
                    generate_img = x[i].view(3, 32, 32).cpu().data.numpy().transpose(1, 2, 0)
                    generate_img = (generate_img * 255).astype(np.uint8)
                    synthesis_data_per_class_list.append(generate_img)
                    synthesis_data_label_per_class_list.append(p)

        synthesis_data_pre_class = Mydatasetpro(synthesis_data_per_class_list, synthesis_data_label_per_class_list,
                                                args["transform"])
        data_clean_threshold = 1.0 if "clean_rate" not in conf.fl_aggregate else conf.fl_aggregate["clean_rate"]
        one_shot_data_per_class = data_clean(dataset=synthesis_data_pre_class, device=device, category=p, threshold=data_clean_threshold)
        one_shot_data_list.append(one_shot_data_per_class)

    # c = torch.arange(0, 10).long().unsqueeze(1).to(device)
    # element_list = [len(data) for data in args["total_semi_supervised_data"]]
    # weights = [1 / ((element + 1) ** 2) for element in element_list]
    # total_weight = sum(weights)
    # normalized_weights = [weight / total_weight for weight in weights]
    # for p, weight in enumerate(normalized_weights):
    #     j = 0
    #     sample_per_class = int(weight * synthesis_sample)
    #     print("class:", p, ", sample_per_class:", sample_per_class)
    #     while j < sample_per_class:
    #         z = torch.randn([c.size(0), latent_size]).to(device)
    #         for model_idx, m in enumerate(args["VAE_lst"]):
    #             x = m.inference(z, c=c)
    #             if "mnist" in args["dataset"]:
    #                 generate_img = x[p].view(28, 28).cpu().data.numpy()
    #             else:
    #                 generate_img = x[p].view(3, 32, 32).cpu().data.numpy().transpose(1, 2, 0)
    #                 generate_img = (generate_img * 255).astype(np.uint8)
    #             synthesis_data_list.append(generate_img)
    #             synthesis_data_label_list.append(p)
    #         j += len(args["VAE_lst"])

    # for model_idx, m in enumerate(args["VAE_lst"]):
    #     for j in range(int(sample_per_model / 10)):
    #         # save_img = True if j % 100 == 1 else False
    #         save_img = False
    #         z = torch.randn([c.size(0), latent_size]).to(device)
    #         x = m.inference(z, c=c)
    #         if save_img:
    #             plt.figure()
    #             plt.figure(figsize=(5, 10))
    #         for p in range(10):
    #             if "mnist" in args["dataset"]:
    #                 generate_img = x[p].view(28, 28).cpu().data.numpy()
    #             else:
    #                 generate_img = x[p].view(3, 32, 32).cpu().data.numpy().transpose(1, 2, 0)
    #                 generate_img = (generate_img * 255).astype(np.uint8)
    #             if save_img:
    #                 plt.subplot(5, 2, p + 1)
    #                 plt.text(0, 0, "c={:d}".format(c[p].item()), color='black',
    #                          backgroundcolor='white', fontsize=8)
    #                 plt.imshow(generate_img)
    #                 plt.axis('off')
    #             synthesis_data_list.append(generate_img)
    #             synthesis_data_label_list.append(p)
    #         if save_img:
    #             plt.savefig("./synthesis_figs/round-%d-model-%d-%d.png" % (args["current_round"], model_idx, j))

    #         # for p in [5, 5, 6]:
    #         #     if "mnist" in args["dataset"]:
    #         #         generate_img = x[p].view(28, 28).cpu().data.numpy()
    #         #     else:
    #         #         generate_img = x[p].view(3, 32, 32).cpu().data.numpy().transpose(1, 2, 0)
    #         #         generate_img = (generate_img * 255).astype(np.uint8)
    #         #     synthesis_data_list.append(generate_img)
    #         #     synthesis_data_label_list.append(p)

    if conf.one_shot:
        return one_shot_data_list
    else:
        return one_shot_data_list
        # return synthesis_data_list, synthesis_data_label_list


def prepare_client_models(conf, fedavg_models, resnet_model, vae_model):
    _client_models = {}
    for arch, _ in fedavg_models.items():
        if "vae" in arch:
            _client_models[arch] = deepcopy(vae_model).cpu()
        else:
            _client_models[arch] = deepcopy(resnet_model).cpu()

    if conf.one_shot:
        _client_models["global"] = resnet_model.cpu()
    return _client_models


# def visualize_vae(vae_model, device, latent_size, current_round):
#     c = torch.arange(0, 10).long().unsqueeze(1).to(device)
#     z = torch.randn([c.size(0), latent_size]).to(device)
#     x = vae_model.inference(z, c=c)
#     plt.figure()
#     plt.figure(figsize=(5, 10))
#     for p in range(10):
#         generate_img = x[p].view(28, 28).cpu().data.numpy()
#         plt.subplot(5, 2, p + 1)
#         plt.text(0, 0, "c={:d}".format(c[p].item()), color='black',
#                  backgroundcolor='white', fontsize=8)
#         plt.imshow(generate_img)
#         plt.axis('off')
#     plt.savefig("./synthesis_figs/round-%d-global_vae_model.png" % (current_round))


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)


def aggregate(
        conf,
        fedavg_models,
        client_models,
        criterion,
        metrics,
        flatten_local_models,
        test_data_loader,
        total_semi_supervised_data,
        train_global_model=False,
):
    # fl_aggregate = conf.fl_aggregate

    # recover the models on the computation device.
    _, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models
    )

    # include model from previous comm. round.
    # if (
    #     "include_previous_models" in fl_aggregate
    #     and fl_aggregate["include_previous_models"] > 0
    # ):
    #     local_models = agg_utils.include_previous_models(conf, local_models)

    # hyper-parameters
    if conf.data == "mnist":
        epochs = 20
    elif conf.data == "emnist":
        epochs = 30
    elif conf.data == "cifar10":
        epochs = 30
    else:
        epochs = 20
    batch_size = 64
    if conf.fedcvae:
        if conf.data == "fashionmnist":
            learning_rate = 5e-4
        elif conf.data == "svhn":
            learning_rate = 2e-4
        elif conf.data == "emnist":
            learning_rate = 5e-5
        else:
            learning_rate = 1e-5
    else:
        if conf.data == "mnist":
            learning_rate = 1e-5
        elif conf.data == "fashionmnist":
            learning_rate = 5e-4
        elif conf.data == "svhn":
            learning_rate = 5e-5
        elif conf.data == "emnist":
            learning_rate = 5e-5
        else:
            learning_rate = 1e-4
    synthesis_sample = conf.num_synthesis_sample
    synthesis_pool_each_class = 300
    latent_size = conf.latent_size
    device = torch.device("cuda") if conf.graph.on_cuda else torch.device("cpu")
    # if conf.data == "mnist":
    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #     test_dataloader = torch.utils.data.DataLoader(
    #         datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    #         batch_size=batch_size, shuffle=False)
    # elif conf.data == "fashionmnist":
    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    #     test_dataloader = torch.utils.data.DataLoader(
    #         datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    #         batch_size=batch_size, shuffle=False)
    # elif conf.data == "cifar10":
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    #     test_dataloader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor()),
    #         batch_size=batch_size, shuffle=False)
    # elif conf.data == "svhn":
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     test_dataloader = torch.utils.data.DataLoader(
    #         datasets.SVHN('./data', split='test', download=True, transform=transforms.ToTensor()),
    #         batch_size=batch_size, shuffle=False)

    transform = transforms.Compose([transforms.ToTensor()])

    # get resnet model prototype (global model)
    resnet_model, vae_model = deepcopy(get_resnet_vae_model(conf, fedavg_models))
    resnet_model, vae_model = resnet_model.to(device), vae_model.to(device)

    # get VAE model list
    num_class = 47 if conf.data == "emnist" else 10
    VAE_lst, VAE_data_partition_lst = get_VAE_list(num_class, local_models)
    if len(VAE_lst) == 0:
        return prepare_client_models(conf, fedavg_models, resnet_model, vae_model), total_semi_supervised_data

    # get synthesis samples from each local model and global model
    # visualize_vae(vae_model, device, latent_size, conf.graph.comm_round)

    # get synthesis data
    one_shot_data_list = get_synthesis_data_list(conf, synthesis_sample,
    latent_size, device, from_pertrained=False, VAE_lst=VAE_lst,
    VAE_data_partition_lst=VAE_data_partition_lst, current_round=conf.graph.comm_round, dataset=conf.data,
    total_semi_supervised_data=total_semi_supervised_data, transform=transform)
    # print("get %d synthesis_data" % len(synthesis_data_list))

    # get dataloader
    # labels = [0 for _ in range(len(synthesis_data_list))]
    # synthesis_raw_data = Mydatasetpro(synthesis_data_list, synthesis_data_label_list, transform)
    # synthesis_raw_dataloader = data.DataLoader(synthesis_raw_data, batch_size=64, shuffle=False)
    # print("synthesis_raw_dataloader done")

    # test synthesis samples from the fedavg vae model
    # fedavg_synthesis_data_list, fedavg_synthesis_data_label_list = [], []
    # c = torch.arange(0, 10).long().unsqueeze(1).to(device)
    # for j in range(100):
    #     z = torch.randn([c.size(0), latent_size]).to(device)
    #     x = vae_model.inference(z, c=c)
    #     for p in range(10):
    #         generate_img = x[p].view(28, 28).cpu().data.numpy()
    #         fedavg_synthesis_data_list.append(generate_img)
    #         fedavg_synthesis_data_label_list.append(p)
    # fedavg_synthesis_raw_data = Mydatasetpro(fedavg_synthesis_data_list, fedavg_synthesis_data_label_list, transform)
    # fedavg_synthesis_raw_dataloader = data.DataLoader(fedavg_synthesis_raw_data, batch_size=64, shuffle=False)
    # fedavg_prob_book = np.array([0.0 for _ in range(10)])
    # fedavg_prob_cnt_book = np.array([0 for _ in range(10)])
    # resnet_model.eval()
    # for img, label in fedavg_synthesis_raw_dataloader:
    #     pred = resnet_model(img.to(device))
    #     max_prob, pred = torch.max(pred.data, dim=1)
    #     max_prob = max_prob.cpu().detach().numpy()
    #     pred_batch, label_batch = pred.cpu().detach(), label.cpu().detach()
    #
    #     for idx, prob in enumerate(max_prob):
    #         fedavg_prob_book[pred_batch[idx]] += prob
    #         fedavg_prob_cnt_book[pred_batch[idx]] += 1
    # for i, (prob, cnt) in enumerate(zip(fedavg_prob_book, fedavg_prob_cnt_book)):
    #     fedavg_prob_book[i] = prob / cnt if cnt > 0 else 0
    # print("fedavg_prob_book:", fedavg_prob_book)

    # predict on synthesis data
    semi_supervised_data_list, semi_supervised_label = [], []
    semi_supervised_data = [[] for _ in range(10)]
    low_prob, wrong_label = 0, 0
    max_prob_list = []
    class_book = np.array([0 for _ in range(10)])
    prob_book = np.array([0.0 for _ in range(10)])
    prob_cnt_book = np.array([0 for _ in range(10)])

    # resnet_model.eval()
    # for img, label in synthesis_raw_dataloader:
    #     # print(img.shape)   # torch.Size([64, 1, 28, 28])
    #     pred = resnet_model(img.to(device))
    #
    #     # Imitating FixMatch (retain pseudo labels with high confidence)
    #     max_prob, pred = torch.max(pred.data, dim=1)
    #     max_prob = max_prob.cpu().detach().numpy()
    #
    #     img_batch = img.cpu().detach().numpy()
    #     # print("img_batch", img_batch.shape)
    #     # print("img_batch[0]", img_batch[0].shape)
    #     pred_batch, label_batch = pred.cpu().detach(), label.cpu().detach()
    #
    #     for idx, prob in enumerate(max_prob):
    #         # print(prob, label_batch[idx])
    #         prob_book[pred_batch[idx]] += prob
    #         prob_cnt_book[pred_batch[idx]] += 1
    #         if prob > threshold and pred_batch[idx] == label_batch[idx]:
    #             if "mnist" in conf.data:
    #                 img = img_batch[idx][0]
    #             else:
    #                 img = img_batch[idx].transpose(1, 2, 0)
    #                 img = (img * 255).astype(np.uint8)
    #             semi_supervised_data[int(pred_batch[idx])].append(img)
    #             # semi_supervised_data_list.append(img)
    #             # semi_supervised_label.append(pred_batch[idx])
    #             class_book[pred_batch[idx]] += 1
    #         else:
    #             if prob < threshold:
    #                 low_prob += 1
    #                 max_prob_list.append(prob)
    #             else:
    #                 wrong_label += 1
    # for i, (prob, cnt) in enumerate(zip(prob_book, prob_cnt_book)):
    #     prob_book[i] = prob / cnt if cnt > 0 else 0
    # # print("get %d train data" % len(semi_supervised_data_list))
    # print("low_prob:", low_prob, "wrong_label:", wrong_label, "\n", max_prob_list)
    #
    # # maintain the synthesis data pool
    # for i in range(10):
    #     total_semi_supervised_data[i] += semi_supervised_data[i]
    #     if len(total_semi_supervised_data[i]) > synthesis_pool_each_class:
    #         total_semi_supervised_data[i] = total_semi_supervised_data[i][-synthesis_pool_each_class:]
    #
    # if not train_global_model or sum([len(total_semi_supervised_data[i]) for i in range(10)]) == 0:
    #     return prepare_client_models(conf, fedavg_models, resnet_model,
    #                                  vae_model), class_book, prob_book, total_semi_supervised_data

    # train_img += semi_supervised_data_list
    # train_label += semi_supervised_label
    print("traning data each class", [len(total_semi_supervised_data[i]) for i in range(10)])
    if conf.one_shot:
        train_data = ConcatDataset(one_shot_data_list)
    else:
        train_data = ConcatDataset(one_shot_data_list)
        # train_data = Mydatasetpro([element for row in total_semi_supervised_data for element in row],
        #                           [i for i, row in enumerate(total_semi_supervised_data) for _ in row], transform)
    # train_data = Mydatasetpro(train_img, train_label, transform)
    # train_data = Mydatasetpro(semi_supervised_data_list, semi_supervised_label, transform)
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print("train dataloader done")

    # d = datasets.MNIST('./', train=True, download=True, transform=transforms.ToTensor())
    # d.data, d.targets = d.data[:3000], d.targets[:3000]
    # scratch_train_dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True)

    # semi-supervised training
    # optimizer_global_model = optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_global_model = optim.Adam(resnet_model.parameters(), lr=learning_rate)


    # scratch_model = ResNet8(10)
    scratch_model = VGG9("emnist")
    scratch_model = scratch_model.to(device)
    # optimizer_scratch_model = optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_scratch_model = optim.Adam(scratch_model.parameters(), lr=learning_rate)
    loss_criterion_ce = torch.nn.CrossEntropyLoss()
    loss_criterion_kd = torch.nn.KLDivLoss()

    # local model as teacher models
    if conf.is_md:
        teacher_models = []
        for t in list(local_models.values()):
            if t.name not in ["vae", "VAE", "cvae", "cvae_mnist", "cvae_cifar", "cvae_large"]:
                teacher_models.append(t)
    # self-distillation, prevent catastrophic forgetting
    if conf.is_sd:
        teacher_models = [deepcopy(resnet_model)]

    print("begin training global model with synthesis data...")
    best_acc = 0
    acc_list = []
    for epoch in range(epochs):
        resnet_model.train()
        train_loss = 0
        for batch_idx, img_target in enumerate(train_dataloader):
            img, target = img_target
            img, target = img.to(device), target.to(device)
            # print("------------model and data location------------")
            # print(img.device, target.device, next(resnet_model.parameters()).device)

            pred = resnet_model(img)
            loss = loss_criterion_ce(pred, target)


            # Forward pass through teacher models to get soft labels
            if conf.is_sd or conf.is_md:
                soft_labels = torch.zeros_like(resnet_model(img), requires_grad=False).cuda()
                for teacher_model in teacher_models:
                    teacher_model.eval()
                    with torch.no_grad():
                        soft_labels += teacher_model(img)
                soft_labels /= len(teacher_models)
                loss_kd = loss_criterion_kd(F.log_softmax(pred, dim=1), F.softmax(soft_labels, dim=1))
                loss = loss + loss_kd

            optimizer_global_model.zero_grad()
            loss.backward()
            optimizer_global_model.step()

            train_loss += loss.item()
            if (batch_idx + 1) % 300 == 0:
                print('[%d, %5d] loss: .3%f' % (epoch + 1, batch_idx + 1, train_loss / 300))

        resnet_model.eval()
        correct, total = 0, 0
        for batch_idx, img_target in enumerate(test_data_loader):
            img, target = img_target
            img, target = img.to(device), target.to(device)
            pred = resnet_model(img)

            _, predicted = torch.max(pred.data, dim=1)
            total += target.size(0)
            correct += torch.eq(predicted, target).sum().item()
        acc = 100 * correct / total
        acc_list.append(acc)
        print(
            "epoch %d train loss: %.3f |||||| test accuracy: %.3f %% " % (epoch + 1, train_loss, acc))
        best_acc = max(acc, best_acc)
    print("best test accuracy: %.3f %%" % best_acc)

    print(acc_list)

    # print("begin training the scratch model with synthesis data...")
    # for epoch in range(epochs):
    #     scratch_model.train()
    #     train_loss = 0
    #     for batch_idx, img_target in enumerate(train_dataloader):
    #         img, target = img_target
    #         img, target = img.to(device), target.to(device)
    #
    #         pred = scratch_model(img)
    #         loss = loss_criterion_ce(pred, target)
    #         optimizer_scratch_model.zero_grad()
    #         loss.backward()
    #         optimizer_scratch_model.step()
    #
    #         train_loss += loss.item()
    #
    #     scratch_model.eval()
    #     correct, total = 0, 0
    #     for batch_idx, img_target in enumerate(test_data_loader):
    #         img, target = img_target
    #         img = img.to(device)
    #         target = target.to(device)
    #         pred = scratch_model(img)
    #
    #         _, predicted = torch.max(pred.data, dim=1)
    #         total += target.size(0)
    #         correct += torch.eq(predicted, target).sum().item()
    #     print("train loss: %.3f |||||| test accuracy: %.3f %% " % (train_loss, 100 * correct / total))

    # prepare client models
    return prepare_client_models(conf, fedavg_models, resnet_model,
                                 vae_model), class_book, prob_book, total_semi_supervised_data
