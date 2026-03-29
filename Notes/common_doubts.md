# Common Issues & Doubts (VAE / Auto-Encoding Variational Bayes)

Central references: Kingma & Welling (2013), arXiv:1312.6114.  
This file collects recurring questions from study notes (`doubts.txt`, `vae_paper_notes.md`, `vae_mathematics.md`, `vae_implementation_notes.md`) and a few **implementation pitfalls**.

---

## 1. Why is \(q_\phi(z \mid x)\) Gaussian? Is the loss always Gaussian?

**Short answer:** Gaussian \(q_\phi\) is a **design choice**, not a theorem.

**Reasons it is common:**

- Fully described by mean and (diagonal) variance — easy to output from a network.  
- Smooth and everywhere differentiable — good for gradient-based learning.  
- \(\mathrm{KL}(q_\phi \,\|\, \mathcal{N}(0,I))\) has a **closed form** for diagonal Gaussians — cheap and stable.  

**Alternatives:** Laplace, mixture of Gaussians, normalizing flows, etc. The **ELBO** structure stays; only the difficulty of \(q\), sampling, and KL changes.

**“Is the loss always Gaussian?”**

- The **ELBO** always has a reconstruction-type term \(\mathbb{E}_q[\log p_\theta(x \mid z)]\) and a **KL** (or similar) term.  
- What changes is the **model** \(p_\theta(x \mid z)\):  
  - **Gaussian / MSE** ↔ continuous pixels.  
  - **Bernoulli / cross-entropy** ↔ binary or sigmoid pixels.  
  - Other members of the exponential family match other log-likelihoods.  

So: **ELBO is general**; **Gaussian assumptions** are the usual first tutorial choice for \(q\) and often for \(p(x \mid z)\) on normalized continuous images.

---

## 2. Why \(\log p(x)\) for “quality” or training?

**Training:** We maximize **average** \(\log p(x)\) over the dataset — standard **maximum likelihood**. Using \(\log\) is monotonic with \(p(x)\), turns products into sums, and improves numerical stability.

**Evaluation:** Higher \(\log p(x)\) on **held-out** data usually means the model assigns more density there — a principled generative metric when the likelihood is meaningful. In practice, likelihood can be hard to compare across model classes; people also use FID, samples, etc.

It is **not** “only for neural networks”; MLE is generic.

---

## 3. Difference between \(\theta\) and \(\phi\)

| Symbol | Role | Typical net |
|--------|------|-------------|
| \(\phi\) | **Encoder** / inference network: \(x \mapsto\) parameters of \(q_\phi(z \mid x)\) | “Recognition” model |
| \(\theta\) | **Decoder** / generative network: \(z \mapsto\) parameters of \(p_\theta(x \mid z)\) | “Generative” model |

Both are learned **jointly** by maximizing the ELBO (minimizing negative ELBO + recon/KL as coded).

---

## 4. Where does \(p(z)\) come from?

\(p(z)\) is the **prior** in the **generative story**: we imagine data is produced by first drawing \(z\), then \(x\). It is **chosen by the modeler**.

- \(\mathcal{N}(0,I)\) is standard: simple, symmetric, and pairs well with **analytic KL** against \(q_\phi\).  
- Without a fixed prior, the encoder could spread codes arbitrarily; generation by sampling \(z \sim p(z)\) would break or become meaningless.

---

## 5. How does optimizing the ELBO relate to \(\log p(x)\)?

Always:

\[
\log p(x) \ge \text{ELBO}(x).
\]

Maximizing the **ELBO** pushes up a **certified lower bound** on \(\log p(x)\). The **tightness** of the bound depends on how close \(q_\phi(z \mid x)\) is to the true \(p(z \mid x)\). Better encoder \(\Rightarrow\) smaller gap \(\Rightarrow\) ELBO closer to true \(\log p(x)\).

---

## 6. Reparameterization: why \(z = \mu + \sigma \odot \epsilon\), \(\epsilon \sim \mathcal{N}(0,I)\)?

**Location-scale property:** If \(\epsilon \sim \mathcal{N}(0,I)\), then \(\mu + \sigma \odot \epsilon \sim \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))\) (elementwise for diagonal case).

So **same distribution** as “sample Gaussian with mean \(\mu\) and std \(\sigma\)”, but randomness is **isolated in \(\epsilon\)**. Gradients w.r.t. \(\mu,\sigma\) (and thus \(\phi\)) flow through the deterministic path \(\mu + \sigma \odot \epsilon\).

Direct sampling \(z \sim \mathcal{N}(\mu,\sigma^2)\) as a **non-differentiable** node blocks gradients through \(\mu,\sigma\) in the naive form.

---

## 7. Jensen’s inequality — what’s going on in the ELBO?

**Statement (concave \(\log\)):** \(\log(\mathbb{E}[Y]) \ge \mathbb{E}[\log Y]\).

In the variational argument, we rewrite \(\log p(x)\) using an expectation under \(q\), then apply Jensen to **lower-bound** \(\log p(x)\) by an expectation of \(\log\)-terms — yielding the ELBO form. Intuition: \(\log\) of an average dominates the average of logs; that inequality direction is what produces a **bound** instead of an equality.

**Tiny numeric sanity check:** for values \(1\) and \(100\), \(\log\) of average \(\approx 3.92\), average of \(\log\) \(\approx 2.3\); \(\log\) of average is larger.

---

## 8. Why is the marginal integral “impossible” for neural \(p_\theta(x \mid z)\)?

\(p(x) = \int p_\theta(x \mid z) p(z)\,dz\) has a **closed form** only for special (conjugate) pairs. A deep **nonlinear** \(p_\theta(x \mid z)\) destroys closed-form integration: you cannot write the integral as a simple formula in \(\theta\). Hence **approximate inference** (VI, MCMC, etc.); VAEs use **amortized** VI with a neural \(q_\phi\).

---

## 9 Implementation / code doubts (from study notes)

### 9.1 Why `2 * latent_dim` outputs in the encoder?

A diagonal Gaussian needs **mean** and **log-variance** per dimension → **`latent_dim` + `latent_dim`** numbers, concatenated then split in `forward`.

### 9.2 Why `log_var` instead of \(\sigma\)?

Networks can output any real number; \(\sigma^2\) must be positive. Predicting \(\log \sigma^2\) gives \(\sigma = \sqrt{\exp(\log\_\text{var})}\) always positive.

### 9.3 `sample=True` vs `sample=False`

- **`True`:** use reparameterized random \(z\) — usual for **training** recon term (stochastic).  
- **`False`:** \(z = \mu\) — often used for **stable** reconstructions or plotting latent **means**.

### 9.4 Prior sampling vs posterior sampling

- **Prior:** \(z \sim \mathcal{N}(0,I)\), decode → **unconditional** generation.  
- **Posterior:** encode \(x\) to get \(q_\phi(z\mid x)\), sample \(z\), decode → **stochastic** reconstructions / local diversity around \(x\).

### 9.5 What does `postprocess_image` do?

Inverts dataset normalization, clips to valid range, converts to `uint8` for display. Belongs with the **dataset** class because it knows mean/std used at load time.

### 9.6 Autoencoder vs VAE (one line each)

- **AE:** MSE recon, single deterministic code — **no** guaranteed generative latent geometry.  
- **VAE:** recon + KL to prior — latents encouraged to match \(p(z)\), enabling **sampling**.

---

## 10. KL formula vs code (watch for typos)

Standard diagonal KL to \(\mathcal{N}(0,I)\) uses (per dimension, inside the sum):

\[
-\tfrac{1}{2}\bigl(1 + \log \sigma^2 - \sigma^2 - \mu^2\bigr)
= -\tfrac{1}{2}\bigl(1 + \texttt{log\_var} - e^{\texttt{log\_var}} - \mu^2\bigr).
\]

Some codebases accidentally introduce an **extra factor** on the `log_var` term (e.g. dividing `log_var` by 2 inside the parentheses). That **does not** match the standard closed form. If results look off, **diff your KL term** against the textbook formula above.

*(Raised in `vae_implementation_notes.md`.)*

---

## 11. Known limitations (conceptual)

- **Blurry samples** often appear with Gaussian\MSE decoders — a known VAE phenomenon; not necessarily a bug.  
- Very small `latent_dim` (e.g. 2 for plotting) **hurts** fidelity; it is a visualization compromise.  
- ELBO can be **loose** if \(q_\phi\) is too simple (e.g. factorized Gaussian for complex true posteriors).

 
---

If you add new questions while reading the PDF, append them here under a dated subsection or open a PR-style bullet list so this file stays the single **FAQ**.
