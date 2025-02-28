# README: Reproducing "Reducing Autocorrelation Times in Lattice Simulations with GANs"

## Introduction
This project aims to replicate the results presented in the paper *"Reducing autocorrelation times in lattice simulations with generative adversarial networks" (Pawlowski & Urban, 2020)*. The study explores using Generative Adversarial Networks (GANs) to reduce autocorrelation times in Monte Carlo simulations of lattice field theory, specifically for 2D scalar \(\phi^4\) theory.

We will implement:
- A Monte Carlo simulation for the 2D \(\phi^4\) lattice theory using Hybrid Monte Carlo (HMC).
- A GAN to generate statistically independent field configurations.
- An overrelaxation step using the GAN to enhance sampling efficiency.
- Methods for validating statistical properties and ensuring ergodicity.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch numpy matplotlib tqdm scipy
```

**Hardware Requirements**:
- A GPU is recommended for training GANs (e.g., NVIDIA GPU with CUDA support).
- The original paper used a GTX 1070.

## Directory Structure

```
project_root/
│── data/                        # Directory for storing training and validation datasets
│── models/                      # Saved GAN models
│── results/                     # Output results (autocorrelation plots, observables, etc.)
│── src/
│   ├── phi4_lattice.py          # Implementation of 2D scalar φ4 theory
│   ├── hmc.py                   # Hybrid Monte Carlo (HMC) implementation
│   ├── gan.py                   # GAN architecture and training
│   ├── overrelaxation.py        # Overrelaxation step using GAN
│   ├── validate.py              # Statistical tests for ergodicity and distribution matching
│── train_gan.py                 # Script for training the GAN
│── run_simulation.py            # Main script to run HMC + GAN sampling
│── analyze_results.py           # Post-processing script to compute autocorrelations
│── requirements.txt             # Python dependencies
│── README.md                    # Project documentation (this file)
```

---

## Step 1: Simulating the \(\phi^4\) Lattice Model

The scalar \(\phi^4\) model is simulated on a 2D Euclidean lattice using periodic boundary conditions. The dimensionless action is given by:

\[
S = \sum_{x \in \Lambda} \left[ -2\kappa \sum_{\mu=1}^{d} \phi(x) \phi(x+\hat{\mu}) + (1-2\lambda)\phi(x)^2 + \lambda\phi(x)^4 \right]
\]

where:
- \(\kappa\) is the hopping parameter.
- \(\lambda\) is the quartic coupling.
- The order parameter (magnetization) is:

  \[
  M = \frac{1}{V} \sum_{x \in \Lambda} \phi(x)
  \]

**Implementation**:
- Use Hybrid Monte Carlo (HMC) to generate lattice configurations.
- Measure observables like magnetization \(M\), susceptibility \(\chi_2\), and Binder cumulant \(U_L\).
- Save generated configurations for training the GAN.

Run:

```bash
python run_simulation.py --lattice_size 32 --kappa 0.21 --lambda 0.022 --samples 10000
```

---

## Step 2: Training the GAN

A **vanilla GAN** is used, with:
- **Generator:** Fully connected layers mapping a latent space \( z \) to lattice configurations \( \phi(x) \).
- **Discriminator:** A binary classifier distinguishing real HMC samples from generated ones.

**Training settings**:
- **Input:** HMC-generated configurations.
- **Latent dimension:** \(d_z = 256\) (tunable).
- **Training iterations:** ~few minutes on GPU.

Run:

```bash
python train_gan.py --data_dir data/ --epochs 5000 --latent_dim 256
```

---

## Step 3: GAN Overrelaxation Step

The **overrelaxation step**:
1. Sample a GAN-generated configuration \(G(z)\).
2. Adjust \(z\) via **gradient descent** to match the action \(S[G(z)] \approx S[\phi]\).
3. Accept the new configuration if action differences are negligible.

**Implementation**:
- Pre-sample configurations to ensure \(|\Delta S| < \Delta S_{\text{thresh}}\).
- Perform a gradient descent step to minimize \(\Delta S^2\).
- Accept if \(\Delta S \approx 0\).

Run:

```bash
python overrelaxation.py --gan_model models/gan.pth --hmc_samples data/hmc_samples.npy
```

---

## Step 4: Statistical Validation

To confirm:
- **Action distributions match** HMC samples.
- **Selection probability is symmetric**.
- **No autocorrelation** in non-action observables.

Run:

```bash
python validate.py --samples results/generated_samples.npy
```

---

## Step 5: Measuring Autocorrelations

The key improvement is the reduction of autocorrelation times in the Markov chain.

Run:

```bash
python analyze_results.py --method HMC --method GAN
```

Expected **autocorrelation function** plot:

- **HMC**: Slow decay of \(C_M(t)\).
- **HMC + GAN Overrelaxation**: Near-zero correlation after every step.

---

## Results & Discussion

- **Efficiency gain**: The integrated autocorrelation time \(\tau_{\text{int}}\) drops from **2.29 (HMC)** to **0.75 (HMC + GAN)**.
- **Computational cost**:
  - HMC step: ~42 ms
  - GAN overrelaxation step: ~117 ms
- **Further optimizations**:
  - Precomputed GAN samples for faster rejection sampling.
  - Conditional GANs trained on action values.

---

## Conclusion

This project demonstrates that GAN-enhanced lattice sampling can **significantly reduce autocorrelation times**, making Monte Carlo simulations more efficient.

Future work:
- Test on different lattice sizes and coupling constants.
- Extend to gauge theories (e.g., U(1) lattice gauge theory).
- Implement alternative generative models like normalizing flows.

For questions, open an issue or contact the authors.

