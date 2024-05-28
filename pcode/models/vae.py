# -*- coding: utf-8 -*-

import torch
from torch.nn import functional as F

__all__ = ["vae"]


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
        self.name = "VAE"

    def forward(self, x):  # x: bs,input_size
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.linear(x))  # -> bs,hidden_size
        mu = self.mu(x)  # -> bs,latent_size
        sigma = self.sigma(x)  # -> bs,latent_size
        return mu, sigma


class Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x:bs,latent_size
        x = F.relu(self.linear1(x))  # ->bs,hidden_size
        x = torch.sigmoid(self.linear2(x))  # ->bs,output_size
        return x


class VAE(torch.nn.Module):
    def __init__(self, dataset, latent_size=32, hidden_size=256):
        super(VAE, self).__init__()
        self.img_size = self._decide_size(dataset)
        self.encoder = Encoder(self.img_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, self.img_size)
        self.name = "vae"

    def _decide_size(self, dataset):
        if dataset == "mnist":
            return 28 * 28
        elif dataset == "cifar10" or dataset == "cifar100":
            return 32 * 32 * 3

    def forward(self, x):  # x: bs,input_size
        mu, sigma = self.encoder(x)  # mu,sigma: bs,latent_size
        # sample
        eps = torch.randn_like(sigma)  # eps: bs,latent_size
        z = mu + eps * sigma  # z: bs,latent_size
        re_x = self.decoder(z)  # re_x: bs,output_size
        return re_x, mu, sigma


def vae(conf, arch=None):
    dataset = conf.data
    model = VAE(dataset)

    return model