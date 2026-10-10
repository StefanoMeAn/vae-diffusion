"""VAE losses, training, and checkpoint loading."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .models import ConvVAE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Sum over pixels/latent coordinates; average over images."""
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, x, reduction="none"
    ).flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def run_epoch(model, loader, device, beta=1.0, optimizer=None, generator=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(x, generator=generator)
            losses = vae_loss(logits, x, mu, logvar, beta)
            if not torch.isfinite(losses[0]).item():
                raise FloatingPointError("Non-finite VAE loss; stop and inspect this run.")
            if training:
                losses[0].backward()
                optimizer.step()
            batch_size = x.shape[0]
            totals += np.array([v.detach().item() for v in losses]) * batch_size
            count += batch_size
    if count == 0:
        raise ValueError("The data loader contains no images.")
    return dict(zip(["loss", "reconstruction", "kl"], (totals / count).tolist()))


def fit_vae(model, train_loader, val_loader, device, config, checkpoint_dir, metric_dir):
    checkpoint_dir, metric_dir = Path(checkpoint_dir), Path(metric_dir)
    # A new run must not silently replace an earlier or interrupted experiment.
    for directory in [checkpoint_dir, metric_dir]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Run already has files: {directory}. Load it or choose a new run tag.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    best_loss = float("inf")
    best_epoch = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in tqdm(range(1, config["epochs"] + 1), desc=config["run_id"]):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train = run_epoch(model, train_loader, device, config["beta"], optimizer)
        # The same validation noise stream is used each epoch. It does not
        # consume the training RNG, and the sampled objective stays comparable.
        generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
        val = run_epoch(model, val_loader, device, config["beta"], generator=generator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        row = {"epoch": epoch, "epoch_seconds": seconds,
               **{f"train_{k}": v for k, v in train.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        improved = val["loss"] < best_loss
        if improved:
            best_loss, best_epoch = val["loss"], epoch
        payload = {"model_state": model.state_dict(), "config": config,
                   "epoch": epoch, "best_epoch": best_epoch,
                   "best_val_loss": best_loss}
        torch.save(payload, checkpoint_dir / "last.pt")
        if improved:
            torch.save(payload, checkpoint_dir / "best.pt")
        pd.DataFrame(history).to_csv(metric_dir / "history.csv", index=False)
        tqdm.write(f"Epoch {epoch:02d}: train={train['loss']:.2f}, val={val['loss']:.2f}, "
                   f"recon={val['reconstruction']:.2f}, KL={val['kl']:.2f} ({seconds:.1f}s)")
    summary = {"best_epoch": best_epoch, "best_val_loss": best_loss,
               "train_and_validation_seconds": sum(r["epoch_seconds"] for r in history),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "peak_allocated_MiB": (torch.cuda.max_memory_allocated(device) / 1024**2
                                       if device.type == "cuda" else None)}
    # This file marks a completed run; checkpoints are inference weights,
    # not full optimizer/RNG snapshots for resuming interrupted training.
    (metric_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return pd.DataFrame(history), summary


def load_vae(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = saved["config"]
    model = ConvVAE(config["latent_dim"], config["hidden_dim"]).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved
