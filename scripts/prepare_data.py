"""Download the datasets and save stratified train/validation indices."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def make_split(targets, validation_size=10000, seed=111):
    targets = np.asarray(targets)
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_size, random_state=seed,
        shuffle=True, stratify=targets,
    )
    return train_idx, val_idx


def prepare_dataset(name, dataset_class, seed):
    # The official test partition stays separate from model selection.
    training = dataset_class(root=ROOT / "data/raw", train=True, download=True)
    testing = dataset_class(root=ROOT / "data/raw", train=False, download=True)
    targets = np.asarray(training.targets)
    train_idx, val_idx = make_split(targets, seed=seed)
    split_dir = ROOT / "data/splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{name}_seed{seed}.npz"
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            if not (np.array_equal(saved["train_idx"], train_idx)
                    and np.array_equal(saved["val_idx"], val_idx)):
                raise RuntimeError(f"Existing split differs: {split_path}. Use another seed or inspect it before replacing.")
    else:
        np.savez_compressed(split_path, train_idx=train_idx, val_idx=val_idx)

    metadata = {
        "dataset": name,
        "seed": seed,
        "split_method": "stratified sklearn train_test_split",
        "validation_size": len(val_idx),
        "train_size": len(train_idx),
        "test_size": len(testing),
        "index_source": "torchvision dataset with train=True",
        "test_source": "torchvision dataset with train=False",
        "classes": list(training.classes),
        "train_class_counts": np.bincount(targets[train_idx], minlength=10).tolist(),
        "validation_class_counts": np.bincount(targets[val_idx], minlength=10).tolist(),
        "test_class_counts": np.bincount(np.asarray(testing.targets), minlength=10).tolist(),
    }
    split_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: {len(train_idx):,} train / {len(val_idx):,} validation / {len(testing):,} test")
    print(f"Saved indices: {split_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "all"], default="all")
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    from torchvision.datasets import MNIST, FashionMNIST

    datasets = {"mnist": MNIST, "fashion_mnist": FashionMNIST}
    for name, dataset_class in datasets.items():
        if args.dataset in ("all", name):
            prepare_dataset(name, dataset_class, args.seed)


if __name__ == "__main__":
    main()
