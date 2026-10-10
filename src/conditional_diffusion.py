"""Class-conditioned DDPM using the same 28x28 U-Net as the baseline."""
import torch
from torch import nn
from torch.nn import functional as F
from .diffusion import DDPM, NoiseUNet


class ConditionalNoiseUNet(NoiseUNet):
    """Condition the timestep representation on a class label (0-9).

    Label 10 is reserved for unconditional predictions used by CFG.
    """
    def __init__(self, base_channels=32, time_dim=64, num_classes=10):
        super().__init__(base_channels=base_channels, time_dim=time_dim)
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes + 1, time_dim * 2)

    def forward(self, x, t, labels):
        if labels.shape != (x.shape[0],):
            raise ValueError('labels must have shape (batch,)')
        emb = self.time_embedding(t) + self.class_embedding(labels)
        x1 = self.enc1(x, emb)
        x2 = self.enc2(F.avg_pool2d(x1, 2), emb)
        h = self.middle(F.avg_pool2d(x2, 2), emb)
        h = F.interpolate(h, size=x2.shape[-2:], mode='nearest')
        h = self.dec2(torch.cat([h, x2], dim=1), emb)
        h = F.interpolate(h, size=x1.shape[-2:], mode='nearest')
        h = self.dec1(torch.cat([h, x1], dim=1), emb)
        return self.output(h)


class ConditionalDDPM(DDPM):
    def loss(self, model, x0, labels, drop_prob=0.1):
        t = torch.randint(self.timesteps, (x0.shape[0],), device=x0.device)
        xt, noise = self.add_noise(x0, t)
        labels = labels.clone()
        labels[torch.rand(labels.shape, device=labels.device) < drop_prob] = model.num_classes
        return F.mse_loss(model(xt, t, labels), noise)

    @torch.no_grad()
    def sample(self, model, labels, device, guidance_scale=2.0, shape=(1, 28, 28)):
        """CFG: eps = eps_uncond + scale * (eps_cond - eps_uncond)."""
        model.eval()
        labels = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)
        if labels.numel() == 0 or ((labels < 0) | (labels >= model.num_classes)).any():
            raise ValueError('Provide class labels between 0 and num_classes-1')
        x = torch.randn((len(labels), *shape), device=device)
        null_labels = torch.full_like(labels, model.num_classes)
        for step in reversed(range(self.timesteps)):
            t = torch.full_like(labels, step)
            eps_cond = model(x, t, labels)
            if guidance_scale == 1:
                eps = eps_cond
            else:
                eps_uncond = model(x, t, null_labels)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            beta = self.extract(self.betas, t, x)
            alpha = self.extract(self.alphas, t, x)
            alpha_bar = self.extract(self.alpha_bar, t, x)
            mean = (x - beta / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()
            if step:
                x = mean + self.extract(self.posterior_variance, t, x).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return (x.clamp(-1, 1) + 1) / 2
