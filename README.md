# VAE and Diffusion Models on MNIST and Fashion-MNIST

A comparison of three generative approaches for 28 × 28 grayscale images: a convolutional variational autoencoder (VAE), an unconditional denoising diffusion probabilistic model (DDPM), and a class-conditional DDPM. The experiments use MNIST and Fashion-MNIST to examine sample appearance, generation speed, and the effect of class conditioning.

## Project overview

The project starts with dataset exploration and a VAE baseline, then trains an unconditional diffusion model and extends it with class labels and classifier-free guidance. The final notebook compares generated samples, sampling time, and agreement between requested labels and predictions from an evaluation classifier.

## Methods

- **VAE:** A convolutional encoder learns a latent distribution, and a decoder reconstructs or generates images. Latent dimensions of 2 and 20 are evaluated.
- **Unconditional DDPM:** A small U-Net predicts noise added at randomly sampled diffusion steps. New images are generated through iterative denoising from Gaussian noise.
- **Conditional DDPM:** Class embeddings are added to the noise predictor. Classifier-free guidance allows generation to be directed toward a requested digit or clothing category.

Both datasets use consistent train/validation splits across experiments. The model architectures and training utilities are implemented in `src/`, while notebooks document the experiments and their outputs.

## Results

| Model | MNIST best validation loss | Fashion-MNIST best validation loss |
| --- | ---: | ---: |
| VAE, latent dimension 2 | 144.49 | 257.45 |
| VAE, latent dimension 20 | 97.07 | 235.60 |
| Unconditional DDPM | 0.03973 | 0.06933 |
| Conditional DDPM | 0.03953 | 0.07028 |

**Note:** VAE losses combine reconstruction and KL-divergence terms, whereas DDPM losses measure noise-prediction error. These values should not be compared directly across model families.

The VAE generated images substantially faster than the diffusion models in the recorded batch benchmark. The diffusion samples generally showed more defined structures, although some examples remained ambiguous. Class conditioning allowed categories to be requested explicitly.

For the conditional DDPM, an evaluation classifier agreed with the requested class for **98% of MNIST** and **90% of Fashion-MNIST** generated samples (200 images per dataset). The evaluation classifiers achieved **98.59%** and **87.49%** test accuracy, respectively. These are classifier-based agreement scores, not independent measurements of realism or generation accuracy.

### Generated-image comparison

![Generated samples from the three models](results/figures/model_comparison/generated_samples_comparison.png)

> If the comparison figure has a different filename locally, update the image path above to match the file in `results/figures/model_comparison/`.

## Repository structure

```text
notebooks/
  01_data_exploration.ipynb
  02_vae.ipynb
  03_diffusion.ipynb
  04_conditional_diffusion.ipynb
  05_model_comparison.ipynb
src/                  # Model definitions and training utilities
scripts/              # Environment checks and data preparation
results/
  figures/             # Saved visualizations
  metrics/             # Experiment summaries and evaluation tables
environment.yml
requirements.txt
README.md
```

## Setup and usage

From the repository root, create the Conda environment using the provided configuration:

```bash
conda env create -f environment.yml
```

Activate the environment named in `environment.yml`, then prepare the data:

```bash
python scripts/check_environment.py
python scripts/prepare_data.py
```

Open the notebooks in numerical order, starting with `01_data_exploration.ipynb`. The training notebooks save model checkpoints locally; `05_model_comparison.ipynb` expects those checkpoints to be available. Trained weights and downloaded datasets are excluded from version control, while figures and metrics are retained.

## Limitations

The experiments use small architectures, 28 × 28 grayscale images, and a single random seed. Sample appearance was assessed visually, and class consistency was estimated using a classifier rather than a distribution-level metric such as FID. The findings describe these particular implementations and training settings rather than a general ranking of generative models.
