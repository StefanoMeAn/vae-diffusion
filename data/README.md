# Data

Run `python scripts/prepare_data.py` from the project root after installing the environment.
The script downloads MNIST and Fashion-MNIST through torchvision into `data/raw/`.
Image data is downloaded on your machine; it is not included in this starter archive.

For each dataset, the official 60,000-image training partition is split into
50,000 training and 10,000 validation images, stratified by class, using seed 111.
The official 10,000-image test partition is kept separate.

The saved files in `data/splits/` contain:

- `{dataset}_seed111.npz`: `train_idx` and `val_idx`, indexing the dataset loaded with `train=True`.
- `{dataset}_seed111.json`: split settings, class names, and class counts.

Both models will use these same splits. Labels are for stratification,
visualization, and clustering evaluation; the initial models will be unconditional.
Use validation data for choosing settings and test data for final reporting.

No resizing, normalization, or augmentation is applied during download.
The future VAE loader will map pixels to [0, 1]; the diffusion loader will map them
to [-1, 1]. Both start from the same original 28 x 28 grayscale images.

Rerunning the download reuses existing data. Rerunning with the same seed verifies
the saved indices; a different seed creates separate split files. Downloads and
generated split files are excluded from Git.

Sources:

- MNIST: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
