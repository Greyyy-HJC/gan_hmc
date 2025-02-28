#!/usr/bin/env python
"""
Run GAN-based overrelaxation for the 2D scalar φ4 theory.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
from datetime import datetime

from src.phi4_lattice import Phi4Lattice
from src.hmc import HMC
from src.gan import LatticeGAN
from src.overrelaxation import GANOverrelaxation

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run GAN-based overrelaxation')
    
    parser.add_argument('--lattice_size', type=int, default=32,
                        help='Lattice size (L x L)')
    parser.add_argument('--kappa', type=float, default=0.21,
                        help='Hopping parameter')
    parser.add_argument('--lambda', type=float, dest='lamb', default=0.022,
                        help='Quartic coupling')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--gan_model', type=str, default='models/gan_latest.pth',
                        help='Path to trained GAN model')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension for GAN (default: 256)')
    parser.add_argument('--hmc_samples', type=str, default=None,
                        help='Path to HMC samples for initial configuration')
    parser.add_argument('--action_threshold', type=float, default=0.1,
                        help='Maximum allowed action difference')
    parser.add_argument('--n_attempts', type=int, default=30,
                        help='Number of attempts to find a suitable configuration')
    parser.add_argument('--n_opt_steps', type=int, default=100,
                        help='Number of optimization steps for action matching')
    parser.add_argument('--hmc_frequency', type=int, default=10,
                        help='Frequency of HMC steps')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save samples')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def load_hmc_samples(hmc_samples_path=None):
    """
    Load HMC samples for initial configuration.
    
    Args:
        hmc_samples_path (str, optional): Path to HMC samples
        
    Returns:
        numpy.ndarray: Loaded samples
    """
    if hmc_samples_path is None:
        # Find the latest samples file
        sample_files = sorted(glob(os.path.join('data', "hmc_samples_*.npy")))
        if not sample_files:
            return None
        hmc_samples_path = sample_files[-1]
    
    print(f"Loading HMC samples from {hmc_samples_path}")
    samples = np.load(hmc_samples_path)
    
    return samples

def main():
    """Run GAN-based overrelaxation."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize lattice
    print(f"Initializing {args.lattice_size}x{args.lattice_size} lattice with κ={args.kappa}, λ={args.lamb}")
    lattice = Phi4Lattice(args.lattice_size, args.kappa, args.lamb)
    
    # Initialize HMC sampler
    hmc = HMC(lattice)
    
    # Load trained GAN
    print(f"Loading GAN model from {args.gan_model}")
    
    # 添加错误处理，确保模型文件存在
    try:
        if not os.path.exists(args.gan_model):
            print(f"Warning: GAN model file not found at {args.gan_model}")
            
            # 尝试查找其他可用的模型文件
            model_files = sorted(glob(os.path.join('models', "gan_*.pth")))
            if model_files:
                alternative_model = model_files[-1]
                print(f"Using alternative model: {alternative_model}")
                args.gan_model = alternative_model
            else:
                raise FileNotFoundError(f"No GAN model files found in 'models' directory")
        
        # 首先尝试加载模型以获取潜在维度信息
        try:
            # 使用 torch.load 获取模型信息，包括 latent_dim
            # 注意：我们使用 weights_only=False 是因为需要加载完整的模型信息，包括 latent_dim
            # 这会产生 FutureWarning，但对于我们的用例是必要的
            checkpoint_info = torch.load(args.gan_model, map_location='cpu', weights_only=False)
            if 'latent_dim' in checkpoint_info:
                latent_dim = checkpoint_info['latent_dim']
                print(f"Detected latent dimension from model: {latent_dim}")
                
                # 如果命令行参数中指定了不同的潜在维度，发出警告
                if args.latent_dim != latent_dim:
                    print(f"Warning: Command line latent_dim ({args.latent_dim}) differs from model latent_dim ({latent_dim})")
                    print(f"Using model's latent_dim: {latent_dim}")
            else:
                latent_dim = args.latent_dim
                print(f"No latent dimension found in model, using command line value: {latent_dim}")
            
            # 使用检测到的潜在维度初始化 GAN
            gan = LatticeGAN(args.lattice_size, latent_dim=latent_dim, device=args.device)
            gan.load_model(args.gan_model)
        except Exception as e:
            print(f"Error detecting latent dimension: {e}")
            print(f"Falling back to default initialization with latent_dim={args.latent_dim}...")
            gan = LatticeGAN(args.lattice_size, latent_dim=args.latent_dim, device=args.device)
            gan.load_model(args.gan_model)
    except Exception as e:
        print(f"Error loading GAN model: {e}")
        print("Please train a GAN model first using train_gan.py")
        exit(1)
    
    # Initialize GAN overrelaxation
    gan_overrelaxation = GANOverrelaxation(
        lattice,
        gan,
        action_threshold=args.action_threshold,
        n_attempts=args.n_attempts,
        n_opt_steps=args.n_opt_steps
    )
    
    # Load HMC samples for initial configuration
    hmc_samples = load_hmc_samples(args.hmc_samples)
    initial_config = hmc_samples[-1] if hmc_samples is not None else None
    
    # Run GAN overrelaxation chain
    samples, observables = gan_overrelaxation.run_chain(
        args.samples,
        initial_config=initial_config,
        hmc_sampler=hmc,
        hmc_frequency=args.hmc_frequency
    )
    
    # Save samples and observables
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_file = os.path.join(args.output_dir, f"gan_samples_{timestamp}.npy")
    observables_file = os.path.join(args.output_dir, f"gan_observables_{timestamp}.npz")
    
    np.save(samples_file, samples)
    np.savez(observables_file, **observables)
    
    print(f"Saved {len(samples)} samples to {samples_file}")
    print(f"Saved observables to {observables_file}")
    
    # Plot observables
    plot_observables(observables, args.lattice_size, args.kappa, args.lamb)
    
    return samples, observables

def plot_observables(observables, L, kappa, lamb):
    """Plot the observables from the GAN overrelaxation."""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot magnetization
    axs[0, 0].plot(observables['magnetization'])
    axs[0, 0].set_title('Magnetization')
    axs[0, 0].set_xlabel('MC step')
    axs[0, 0].set_ylabel('M')
    
    # Plot energy
    axs[0, 1].plot(observables['energy'])
    axs[0, 1].set_title('Energy density')
    axs[0, 1].set_xlabel('MC step')
    axs[0, 1].set_ylabel('E')
    
    # Plot susceptibility
    axs[1, 0].plot(observables['susceptibility'])
    axs[1, 0].set_title('Susceptibility')
    axs[1, 0].set_xlabel('MC step')
    axs[1, 0].set_ylabel('χ')
    
    # Plot Binder cumulant
    axs[1, 1].plot(observables['binder_cumulant'])
    axs[1, 1].set_title('Binder cumulant')
    axs[1, 1].set_xlabel('MC step')
    axs[1, 1].set_ylabel('U_L')
    
    # Add overall title
    plt.suptitle(f'GAN Overrelaxation Observables (L={L}, κ={kappa}, λ={lamb})')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/gan_observables.png')
    plt.close()
    
    # Plot histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnetization histogram
    axs[0, 0].hist(observables['magnetization'], bins=50)
    axs[0, 0].set_title('Magnetization Distribution')
    axs[0, 0].set_xlabel('M')
    axs[0, 0].set_ylabel('Frequency')
    
    # Energy histogram
    axs[0, 1].hist(observables['energy'], bins=50)
    axs[0, 1].set_title('Energy Distribution')
    axs[0, 1].set_xlabel('E')
    axs[0, 1].set_ylabel('Frequency')
    
    # Susceptibility histogram
    axs[1, 0].hist(observables['susceptibility'], bins=50)
    axs[1, 0].set_title('Susceptibility Distribution')
    axs[1, 0].set_xlabel('χ')
    axs[1, 0].set_ylabel('Frequency')
    
    # Binder cumulant histogram
    axs[1, 1].hist(observables['binder_cumulant'], bins=50)
    axs[1, 1].set_title('Binder Cumulant Distribution')
    axs[1, 1].set_xlabel('U_L')
    axs[1, 1].set_ylabel('Frequency')
    
    # Add overall title
    plt.suptitle(f'GAN Overrelaxation Observable Distributions (L={L}, κ={kappa}, λ={lamb})')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/gan_histograms.png')
    plt.close()

if __name__ == '__main__':
    main() 