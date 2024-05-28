import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["cvae"]


class CVAE_MNIST(nn.Module):

    def __init__(self, dataset, latent_size=2, conditional=True, num_labels=10):
        super().__init__()
        if conditional:
            assert num_labels > 0
        # assert type(latent_size) == int

        self.name = "cvae"
        self.latent_size = latent_size
        self.dataset = dataset
        self.num_features = 784 if "mnist" in self.dataset else 3072
        self.encoder_layer_sizes = [self.num_features, 256]
        self.decoder_layer_sizes = [256, self.num_features]
        self.encoder = Encoder(
            self.encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            self.decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):
        if x.dim() > 2:
            if "mnist" in self.dataset:
                x = x.view(-1, 28 * 28)
            else:
                # x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
                x = x.view(-1, 3 * 32 * 32)
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)
        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            c = self.idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

    def idx2onehot(self, idx, n):
        assert torch.max(idx).item() < n
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        onehot = torch.zeros(idx.size(0), n).to(idx.device)
        onehot.scatter_(1, idx, 1)

        return onehot


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()

        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):
        if self.conditional:
            c = self.idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x

    def idx2onehot(self, idx, n):
        assert torch.max(idx).item() < n
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        onehot = torch.zeros(idx.size(0), n).to(idx.device)
        onehot.scatter_(1, idx, 1)

        return onehot


class CVAE_CIFAR(nn.Module):
    def __init__(self, hidden_size=256, z_size=20, num_classes=10):
        super(CVAE_CIFAR, self).__init__()

        self.name = "cvae"
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(256 * 4 * 4, hidden_size),
        #     nn.ReLU()
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, hidden_size),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_size, z_size)
        self.fc_logvar = nn.Linear(hidden_size, z_size)

        # self.decoder = nn.Sequential(
        #     nn.Linear(z_size + num_classes, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 256 * 4 * 4),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (256, 4, 4)),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(z_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Convert class labels to one-hot encoding
        y_onehot = F.one_hot(y, num_classes=10).float().to(x.device)

        # concatenate one-hot encoded class information to the latent space
        z_y = torch.cat([z, y_onehot], dim=1)

        # decode the concatenated representation
        x_recon = self.decoder(z_y)
        return x_recon, mu, logvar



class CVAE_SVHN(nn.Module):
    def __init__(self, hidden_size=256, z_size=20, num_classes=10):
        super(CVAE_SVHN, self).__init__()

        self.name = "cvae"
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(256 * 4 * 4, hidden_size),
        #     nn.ReLU()
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_size),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_size + num_classes, z_size)
        self.fc_logvar = nn.Linear(hidden_size + num_classes, z_size)

        # self.decoder = nn.Sequential(
        #     nn.Linear(z_size + num_classes, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 256 * 4 * 4),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (256, 4, 4)),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(z_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        x = self.encoder(x)
        y_onehot = F.one_hot(y, num_classes=10).float().to(x.device)

        x = torch.cat([x, y_onehot], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        z = self.reparameterize(mu, logvar)

        z_y = torch.cat([z, y_onehot], dim=1)
        x_recon = self.decoder(z_y)
        return x_recon, mu, logvar


class CVAE_EMNIST(nn.Module):
    def __init__(self, input_size=28 * 28, latent_size=20, num_classes=47):
        super(CVAE_EMNIST, self).__init__()

        self.name = "cvae"
        self.num_classes = num_classes

        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(32 * 7 * 7, 256),
        #     nn.ReLU()
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 256),
            nn.ReLU()
        )

        # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_size + num_classes, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 32 * 7 * 7),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (32, 7, 7)),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.mean_layer = nn.Linear(256 + num_classes, latent_size)
        self.logvar_layer = nn.Linear(256 + num_classes, latent_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        x = x.view(-1, 1, 28, 28)
        # y = y - 1
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(x.device)

        # Encoder
        encoded = self.encoder(x)

        # Calculate mean and logvar
        encoded = torch.cat([encoded, y_onehot], dim=1)
        mu = self.mean_layer(encoded)
        logvar = self.logvar_layer(encoded)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoded = self.decoder(torch.cat([z, y_onehot], dim=1))

        return decoded, mu, logvar

    # def forward(self, x, y):
    #     # Encoder
    #     x = x.view(-1, 28 * 28)
    #     # y = y - 1
    #     y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(x.device)
    #     encoder_input = torch.cat([x, y_onehot], dim=1)
    #     # x = self.relu(self.linear1(encoder_input))
    #     # x = self.relu(self.linear2(x))
    #     # x = self.relu(self.linear3(x))
    #     # enc_output = self.sigmoid(self.linear4(x))
    #     enc_output = self.encoder(encoder_input)
    #
    #     # Sampling
    #     mu, logvar = enc_output.chunk(2, dim=1)
    #     z = self.reparameterize(mu, logvar)
    #
    #     # Decoder
    #     decoder_input = torch.cat([z, y_onehot], dim=1)
    #     # x = self.relu(self.linear5(decoder_input))
    #     # x = self.relu(self.linear6(x))
    #     # x = self.relu(self.linear7(x))
    #     # dec_output = self.sigmoid(self.linear8(x))
    #     dec_output = self.decoder(decoder_input)
    #
    #     return dec_output, mu, logvar


def cvae(conf, arch=None):
    dataset = conf.data
    if dataset == "emnist":
        model = CVAE_EMNIST(input_size=28 * 28, latent_size=conf.latent_size, num_classes=47)
    elif "mnist" in dataset:
        model = CVAE_MNIST(dataset, conf.latent_size)
    elif dataset == "svhn":
        model = CVAE_SVHN(z_size=conf.latent_size)
    else:
        model = CVAE_CIFAR(z_size=conf.latent_size)

    return model
