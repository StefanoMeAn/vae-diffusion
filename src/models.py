"""Convolutional VAE for 28 x 28 grayscale images."""

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                     # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                     # 64 x 7 x 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 64 * 7 * 7),
            nn.LeakyReLU(0.2),
        )
        # Upsampling depends only on z, not on encoder pooling indices.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),                    # 32 x 14 x 14
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )                                       # 1 x 28 x 28 logits

    def encode(self, x):
        h = self.encoder(x)
        return self.mean_layer(h), self.logvar_layer(h)

    @staticmethod
    def reparameterize(mu, logvar, generator=None):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn(std.shape, device=std.device, dtype=std.dtype,
                              generator=generator)
        return mu + std * epsilon

    def decode_logits(self, z):
        h = self.decoder_input(z).reshape(-1, 64, 7, 7)
        return self.decoder(h)

    def decode(self, z):
        """Return pixel probabilities for plotting or generation."""
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        return self.decode_logits(z), mu, logvar
