# Variational Autoencoder (VAE) Experiments

This repository is a **single-notebook educational implementation** of autoencoders and variational autoencoders in PyTorch. It walks from a deterministic autoencoder on MNIST through MLP-based and CNN-based VAEs, including experiments on **MNIST**, **Fashion-MNIST**, and **CIFAR-10**.

Original Paper:  `https://arxiv.org/pdf/1312.6114`
Explnation : `<meduium_link>`

The primary artifact is [`VAE.ipynb`](./VAE.ipynb).

## What this project contains

1. **Data pipelines**  
   - Custom `Dataset` wrappers that return **images only** for unsupervised training (labels are kept via helper methods for scatter plots).  
   - **Pixel standardization** using fixed mean/std (MNIST/Fashion-MNIST share the same scalars in the notebook; CIFAR-10 uses standard per-channel stats).  
   - **Postprocessing** helpers to map model outputs back to displayable `uint8` images.

2. **Deterministic autoencoder (AE)**  
   - MLP encoder/decoder on `1×28×28` grayscale inputs.  
   - Default **2-D latent space** for easy 2-D scatter plots of digit classes.  
   - Training with **MSE reconstruction loss**, Adam, and a step learning-rate scheduler.

3. **Variational autoencoder (VAE)**  
   - Shared abstract class: encoder produces **Gaussian parameters** \(μ\) and **log-variance** (\(\log σ^2\)) for a diagonal approximate posterior \(q(z \mid x)\).  
   - **Reparameterization trick** for backprop through stochastic latents.  
   - **ELBO-style loss** implemented as `GaussianELBOLoss`: a reconstruction (likelihood) term plus a **KL** term toward a standard normal prior.  
   - **MLP VAE** for MNIST/Fashion-MNIST (same spatial size as the AE).  
   - **CNN VAE** for `3×32×32` CIFAR-10 (convolutional encoder, transposed-convolutional decoder).

4. **Visualizations** (implemented in the notebook)  
   - Random **reconstruction** grids (original vs decoded).  
   - **Latent space** scatter plots colored by class (AE vs VAE; VAE often uses posterior mean for plotting).  
   - **Sampling from the prior** \(z \sim \mathcal{N}(0, I)\) through the decoder.  
   - **Sampling from the posterior** \(z \sim q(z \mid x)\) for a fixed input (stochastic reconstructions).  
   - **Latent interpolation** on a 2-D grid (for the 2-D latent MLP setup).  
   - Training **history plots** for total loss, likelihood term, and KL term when training the VAE.

## Requirements

The notebook expects a Python environment with:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm` (notebook progress bars)

Install examples:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

Use a PyTorch build that matches your hardware (CPU-only or CUDA). The notebook sets `ACCELERATOR = 'cuda:0'` by default; for CPU-only runs, change that string to `'cpu'` and adjust or remove the small CUDA assertion block at the top of the notebook if it conflicts with your setup.

## How to run

1. Open `VAE.ipynb` in Jupyter, JupyterLab, or VS Code.  
2. Run cells **in order** from the top. Early cells download datasets (MNIST/Fashion-MNIST under `/tmp/mnist`, CIFAR-10 under `/tmp/cifar10`).  
3. Training cells can take noticeable time on CPU; a GPU speeds up the VAE and especially the CIFAR CNN section.

There is no separate CLI or package entry point—everything is orchestrated inside the notebook.

## Notebook structure (high level)

| Section | Focus |
|--------|--------|
| Dataset | MNIST/Fashion-MNIST loading, standardization, `UnsupervisedMNIST` |
| Autoencoder | MLP AE architecture, training loop, reconstructions, 2-D latent scatter |
| VAE | Abstract `VariationalAutoEncoder`, `MLPVariationalAutoEncoder`, `GaussianELBOLoss`, training, plots |
| Demonstrations | Prior/posterior sampling, latent interpolation |
| Fashion-MNIST | Same MLP VAE pipeline on Fashion-MNIST |
| CIFAR-10 | `UnsupervisedCIFAR10`, `CNNBlock` / `TransposeCNNBlock`, `CNNVariationalAutoEncoder`, sampling |

Markdown cells in the notebook also discuss qualitative behavior (e.g. blurry reconstructions, trade-offs between latent dimension and cluster separation, KL regularization pulling latents toward the origin).

## Implementation notes

- **Latent dimension**: The AE and MLP VAE default to **2** latent units so that latent space can be plotted in the plane. This is intentionally small for visualization, not for best distortion or generative quality.  
- **ELBO loss**: The notebook’s `GaussianELBOLoss` combines a scaled squared-error reconstruction term (controlled by `vae_noise`) with an analytic KL term for diagonal Gaussians. Training logs separate **likelihood** and **KL** curves.  
- **CNN VAE**: The CIFAR section uses deeper spatial reasoning than an MLP; the notebook points readers to [D2L: Transposed Convolution](https://d2l.ai/chapter_computer-vision/transposed-conv.html) for background on upsampling with `ConvTranspose2d`.

## Limitations (as discussed in the notebook)

- VAE samples can look **blurred**; this is a known characteristic when using Gaussian observation models and MSE-like reconstructions.  
- **CIFAR-10** is harder than MNIST; qualitative evaluation of prior samples is more difficult.

## Project layout

```
Vae/
├── README.md      # This file
└── VAE.ipynb      # Full implementation, training, and figures
```

## References (concepts)

- Kingma & Welling, *Auto-Encoding Variational Bayes* (VAE / ELBO / reparameterization).  
- Goodfellow et al., *Deep Learning* — autoencoders and generative models.  
- Course-style treatment of transposed convolutions: [D2L transposed convolution](https://d2l.ai/chapter_computer-vision/transposed-conv.html).

---

If you extend this repo, consider adding a `requirements.txt` or `environment.yml` pinned to your tested PyTorch version and documenting any change to `ACCELERATOR` or dataset paths.
