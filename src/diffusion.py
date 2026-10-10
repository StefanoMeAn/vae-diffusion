"""Small time-conditioned noise predictor and DDPM process for 28x28 images."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.layers = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim * 2))

    def forward(self, t):
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        angles = t.float()[:, None] * frequencies[None, :]
        return self.layers(torch.cat([angles.sin(), angles.cos()], dim=-1))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_layer = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, time_embedding):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_layer(time_embedding)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h)))


class NoiseUNet(nn.Module):
    """Two-level U-Net. Input/output: (batch, 1, 28, 28)."""

    def __init__(self, base_channels=32, time_dim=64):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.enc1 = ConvBlock(1, base_channels, time_dim * 2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_dim * 2)
        self.middle = ConvBlock(base_channels * 2, base_channels * 2, time_dim * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels, time_dim * 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, time_dim * 2)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x, t):
        emb = self.time_embedding(t)
        x1 = self.enc1(x, emb)                     # 28 x 28
        x2 = self.enc2(F.avg_pool2d(x1, 2), emb)  # 14 x 14
        h = self.middle(F.avg_pool2d(x2, 2), emb) # 7 x 7
        h = F.interpolate(h, size=x2.shape[-2:], mode='nearest')
        h = self.dec2(torch.cat([h, x2], dim=1), emb)
        h = F.interpolate(h, size=x1.shape[-2:], mode='nearest')
        h = self.dec1(torch.cat([h, x1], dim=1), emb)
        return self.output(h)


class DDPM(nn.Module):
    """Linear-beta DDPM with epsilon prediction; inputs are scaled to [-1, 1]."""

    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if timesteps < 2 or not (0 < beta_start < beta_end < 1):
            raise ValueError('Invalid diffusion schedule')
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        self.timesteps = timesteps
        for name, value in {
            'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar,
            'sqrt_alpha_bar': alpha_bar.sqrt(),
            'sqrt_one_minus_alpha_bar': (1 - alpha_bar).sqrt(),
            'posterior_variance': betas * (1 - alpha_bar_prev) / (1 - alpha_bar),
        }.items():
            self.register_buffer(name, value)

    @staticmethod
    def extract(values, t, x):
        return values[t].reshape(-1, *((1,) * (x.ndim - 1)))

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.extract(self.sqrt_alpha_bar, t, x0) * x0
        xt += self.extract(self.sqrt_one_minus_alpha_bar, t, x0) * noise
        return xt, noise

    def loss(self, model, x0):
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        xt, noise = self.add_noise(x0, t)
        prediction = model(xt, t)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, model, count, device, shape=(1, 28, 28)):
        model.eval()
        x = torch.randn((count, *shape), device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((count,), step, device=device, dtype=torch.long)
            predicted_noise = model(x, t)
            beta = self.extract(self.betas, t, x)
            alpha = self.extract(self.alphas, t, x)
            alpha_bar = self.extract(self.alpha_bar, t, x)
            mean = (x - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                x = mean + self.extract(self.posterior_variance, t, x).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return ((x.clamp(-1, 1) + 1) / 2)
