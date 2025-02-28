#!/usr/bin/env python
"""
Run a Hybrid Monte Carlo (HMC) simulation for the 2D scalar φ4 theory.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from src.phi4_lattice import Phi4Lattice
from src.hmc import HMC

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run HMC simulation for 2D scalar φ4 theory')
    
    parser.add_argument('--lattice_size', type=int, default=32,
                        help='Lattice size (L x L)')
    parser.add_argument('--kappa', type=float, default=0.21,
                        help='Hopping parameter')
    parser.add_argument('--lambda', type=float, dest='lamb', default=0.022,
                        help='Quartic coupling')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--burnin', type=int, default=1000,
                        help='Number of burn-in steps')
    parser.add_argument('--thin', type=int, default=10,
                        help='Thinning factor')
    parser.add_argument('--n_steps', type=int, default=10,
                        help='Number of leapfrog steps in HMC')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Step size for leapfrog integration')
    parser.add_argument('--tune', action='store_true',
                        help='Tune HMC parameters')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save samples')
    
    return parser.parse_args()

def main():
    """Run the HMC simulation."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize lattice
    print(f"Initializing {args.lattice_size}x{args.lattice_size} lattice with κ={args.kappa}, λ={args.lamb}")
    lattice = Phi4Lattice(args.lattice_size, args.kappa, args.lamb)
    
    # Initialize HMC sampler
    hmc = HMC(lattice, args.n_steps, args.step_size)
    
    # Tune HMC parameters if requested
    if args.tune:
        print("Tuning HMC parameters...")
        n_steps, step_size = hmc.tune_parameters()
        print(f"Tuned parameters: n_steps={n_steps}, step_size={step_size:.6f}")
    
    # Run HMC chain
    samples, observables = hmc.run_chain(
        args.samples,
        n_burnin=args.burnin,
        thin=args.thin
    )
    
    # Save samples and observables
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_file = os.path.join(args.output_dir, f"hmc_samples_{timestamp}.npy")
    observables_file = os.path.join(args.output_dir, f"hmc_observables_{timestamp}.npz")
    
    np.save(samples_file, samples)
    np.savez(observables_file, **observables)
    
    print(f"Saved {len(samples)} samples to {samples_file}")
    print(f"Saved observables to {observables_file}")
    
    # Plot observables
    plot_observables(observables, args.lattice_size, args.kappa, args.lamb)
    
    return samples, observables

def plot_observables(observables, L, kappa, lamb):
    """Plot the observables from the HMC simulation."""
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
    plt.suptitle(f'HMC Observables (L={L}, κ={kappa}, λ={lamb})')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/hmc_observables.png')
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
    plt.suptitle(f'HMC Observable Distributions (L={L}, κ={kappa}, λ={lamb})')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/hmc_histograms.png')
    plt.close()

if __name__ == '__main__':
    main() 