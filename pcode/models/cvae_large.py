import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["cvae_large"]


class CVAE(nn.Module):
    def __init__(self, dataset, hidden_size=256, z_size=20, num_classes=10):
        super(CVAE, self).__init__()

        self.name = "cvae_large"
        self.num_classes = num_classes
        self.input_size = 1 if "mnist" in dataset else 3
        self.output_size = 7 if "mnist" in dataset else 8

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * self.output_size * self.output_size, hidden_size),
            nn.ReLU(inplace=True)
        )

        # Latent space layers
        self.fc_mu = nn.Linear(hidden_size + num_classes, z_size)
        self.fc_logvar = nn.Linear(hidden_size + num_classes, z_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_size + num_classes, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 256 * self.output_size * self.output_size),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, self.output_size, self.output_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(64, self.input_size, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        # Encoder
        x = self.encoder(x)

        # Combine with class information
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(x.device)
        x = torch.cat((x, y_onehot), dim=1)

        # Latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        z = torch.cat((z, y_onehot), dim=1)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar


def cvae_large(conf):
    dataset = conf.data
    if dataset == "emnist":
        model = CVAE(dataset=dataset, hidden_size=256, z_size=20, num_classes=47)
    else:
        model = CVAE(dataset=dataset, hidden_size=256, z_size=20, num_classes=10)
    return model
