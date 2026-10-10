"""Training and checkpoint utilities for class-conditioned DDPM experiments."""
import json
import time
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from .conditional_diffusion import ConditionalDDPM, ConditionalNoiseUNet


def run_conditional_epoch(model, process, loader, device, drop_prob=0.1, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True) * 2 - 1
            labels = labels.to(device, non_blocking=True).long()
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = process.loss(model, images, labels, drop_prob=drop_prob)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite conditional diffusion loss')
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
    if not count:
        raise ValueError('Empty data loader')
    return total / count


def fit_conditional(model, process, train_loader, val_loader, device, config, checkpoint_dir, metric_dir):
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
        train_loss = run_conditional_epoch(model, process, train_loader, device,
                                           config['drop_prob'], optimizer)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        try:
            torch.manual_seed(config['seed'] + 1)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(config['seed'] + 1)
            val_loss = run_conditional_epoch(model, process, val_loader, device,
                                             config['drop_prob'])
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'epoch_seconds': time.perf_counter() - start}
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            improved = True
        else:
            improved = False
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


def load_conditional(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = saved['config']
    model = ConditionalNoiseUNet(base_channels=config['base_channels']).to(device)
    model.load_state_dict(saved['model_state'])
    model.eval()
    process = ConditionalDDPM(timesteps=config['timesteps']).to(device)
    return model, process, saved
