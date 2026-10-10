"""Training and checkpoint utilities for the DDPM experiments."""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .diffusion import DDPM, NoiseUNet


def run_diffusion_epoch(model, process, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            images = images.to(device, non_blocking=True) * 2 - 1
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = process.loss(model, images)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite diffusion loss')
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * images.size(0)
            count += images.size(0)
    if count == 0:
        raise ValueError('Empty data loader')
    return total / count


def fit_diffusion(model, process, train_loader, val_loader, device, config,
                  checkpoint_dir, metric_dir):
    checkpoint_dir, metric_dir = Path(checkpoint_dir), Path(metric_dir)
    for directory in (checkpoint_dir, metric_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Run already has files: {directory}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model, process = model.to(device), process.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    history, best_loss, best_epoch = [], float('inf'), None
    for epoch in tqdm(range(1, config['epochs'] + 1), desc=config['run_id']):
        start = time.perf_counter()
        train_loss = run_diffusion_epoch(model, process, train_loader, device, optimizer)
        # Use a fixed RNG stream to compare validation epochs more consistently.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        try:
            torch.manual_seed(config['seed'] + 1)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(config['seed'] + 1)
            val_loss = run_diffusion_epoch(model, process, val_loader, device)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'epoch_seconds': time.perf_counter() - start}
        history.append(row)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
        payload = {'model_state': model.state_dict(), 'config': config,
                   'epoch': epoch, 'best_epoch': best_epoch, 'best_val_loss': best_loss}
        torch.save(payload, checkpoint_dir / 'last.pt')
        if improved:
            torch.save(payload, checkpoint_dir / 'best.pt')
        pd.DataFrame(history).to_csv(metric_dir / 'history.csv', index=False)
        tqdm.write(f"Epoch {epoch:02d}: train={train_loss:.4f}, val={val_loss:.4f}")
    summary = {'best_epoch': best_epoch, 'best_val_loss': best_loss,
               'parameter_count': sum(p.numel() for p in model.parameters())}
    (metric_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return pd.DataFrame(history), summary


def load_diffusion(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = saved['config']
    model = NoiseUNet(base_channels=config['base_channels']).to(device)
    model.load_state_dict(saved['model_state'])
    model.eval()
    process = DDPM(timesteps=config['timesteps']).to(device)
    return model, process, saved
