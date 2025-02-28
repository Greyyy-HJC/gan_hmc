#!/usr/bin/env python
"""
Analyze results from HMC and GAN-based simulations, focusing on autocorrelations.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from src.phi4_lattice import Phi4Lattice
from src.validate import LatticeValidator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze results from simulations')
    
    parser.add_argument('--lattice_size', type=int, default=32,
                        help='Lattice size (L x L)')
    parser.add_argument('--kappa', type=float, default=0.21,
                        help='Hopping parameter')
    parser.add_argument('--lambda', type=float, dest='lamb', default=0.022,
                        help='Quartic coupling')
    parser.add_argument('--hmc_samples', type=str, default=None,
                        help='Path to HMC samples (if None, use latest)')
    parser.add_argument('--gan_samples', type=str, default=None,
                        help='Path to GAN samples (if None, use latest)')
    parser.add_argument('--max_lag', type=int, default=None,
                        help='Maximum lag for autocorrelation calculation')
    parser.add_argument('--observables', type=str, nargs='+',
                        default=['magnetization', 'energy', 'susceptibility', 'binder'],
                        help='Observables to analyze')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def load_latest_samples(sample_type='hmc'):
    """
    Load the latest samples of a given type.
    
    Args:
        sample_type (str): Type of samples ('hmc' or 'gan')
        
    Returns:
        numpy.ndarray: Loaded samples
    """
    # Find the latest samples file
    pattern = f"{sample_type}_samples_*.npy"
    sample_files = sorted(glob(os.path.join('data', pattern)))
    
    if not sample_files:
        raise FileNotFoundError(f"No {sample_type.upper()} samples found in data/")
    
    samples_path = sample_files[-1]
    print(f"Loading {sample_type.upper()} samples from {samples_path}")
    
    return np.load(samples_path)

def main():
    """Analyze results from simulations."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize lattice
    print(f"Initializing {args.lattice_size}x{args.lattice_size} lattice with κ={args.kappa}, λ={args.lamb}")
    lattice = Phi4Lattice(args.lattice_size, args.kappa, args.lamb)
    
    # Initialize validator
    validator = LatticeValidator(lattice)
    
    # Load samples
    try:
        hmc_samples = np.load(args.hmc_samples) if args.hmc_samples else load_latest_samples('hmc')
        gan_samples = np.load(args.gan_samples) if args.gan_samples else load_latest_samples('gan')
        
        print(f"Loaded {len(hmc_samples)} HMC samples and {len(gan_samples)} GAN samples")
        
        # Check lattice size
        if hmc_samples.shape[1:] != (args.lattice_size, args.lattice_size):
            print(f"Warning: HMC samples have shape {hmc_samples.shape}, "
                  f"expected ({args.lattice_size}, {args.lattice_size})")
        
        if gan_samples.shape[1:] != (args.lattice_size, args.lattice_size):
            print(f"Warning: GAN samples have shape {gan_samples.shape}, "
                  f"expected ({args.lattice_size}, {args.lattice_size})")
        
        # Compare action distributions
        print("\nComparing action distributions...")
        validator.compare_action_distributions(hmc_samples, gan_samples)
        
        # Compare observable distributions
        print("\nComparing observable distributions...")
        validator.compare_observable_distributions(hmc_samples, gan_samples)
        
        # Calculate and compare autocorrelations
        print("\nCalculating autocorrelations...")
        
        # Store integrated autocorrelation times
        integrated_times = {
            'hmc': {},
            'gan': {}
        }
        
        for observable in args.observables:
            print(f"\nAnalyzing autocorrelation for {observable}...")
            hmc_tau, gan_tau = validator.compare_autocorrelations(
                hmc_samples, gan_samples, observable, args.max_lag
            )
            
            integrated_times['hmc'][observable] = hmc_tau
            integrated_times['gan'][observable] = gan_tau
        
        # Validate ergodicity
        print("\nValidating ergodicity...")
        validator.validate_ergodicity(hmc_samples, gan_samples)
        
        # Summarize results
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        
        print("\nIntegrated autocorrelation times:")
        print("-"*40)
        print(f"{'Observable':<15} {'HMC':<10} {'GAN+HMC':<10} {'Speedup':<10}")
        print("-"*40)
        
        for observable in args.observables:
            hmc_tau = integrated_times['hmc'][observable]
            gan_tau = integrated_times['gan'][observable]
            speedup = hmc_tau / gan_tau if gan_tau > 0 else float('inf')
            
            print(f"{observable:<15} {hmc_tau:<10.2f} {gan_tau:<10.2f} {speedup:<10.2f}x")
        
        print("\nConclusion:")
        avg_speedup = np.mean([integrated_times['hmc'][obs] / integrated_times['gan'][obs] 
                              for obs in args.observables])
        
        print(f"The GAN-based overrelaxation method achieves an average speedup of {avg_speedup:.2f}x")
        print(f"compared to standard HMC for the 2D scalar φ4 theory with L={args.lattice_size}, "
              f"κ={args.kappa}, λ={args.lamb}.")
        
        # Save summary to file
        summary_file = os.path.join(args.output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SUMMARY OF RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("Integrated autocorrelation times:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Observable':<15} {'HMC':<10} {'GAN+HMC':<10} {'Speedup':<10}\n")
            f.write("-"*40 + "\n")
            
            for observable in args.observables:
                hmc_tau = integrated_times['hmc'][observable]
                gan_tau = integrated_times['gan'][observable]
                speedup = hmc_tau / gan_tau if gan_tau > 0 else float('inf')
                
                f.write(f"{observable:<15} {hmc_tau:<10.2f} {gan_tau:<10.2f} {speedup:<10.2f}x\n")
            
            f.write("\nConclusion:\n")
            f.write(f"The GAN-based overrelaxation method achieves an average speedup of {avg_speedup:.2f}x ")
            f.write(f"compared to standard HMC for the 2D scalar φ4 theory with L={args.lattice_size}, ")
            f.write(f"κ={args.kappa}, λ={args.lamb}.\n")
        
        print(f"\nSummary saved to {summary_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run both HMC and GAN simulations first.")
        return

if __name__ == '__main__':
    main() 