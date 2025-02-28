#!/usr/bin/env python
"""
Train a GAN on HMC-generated samples from the 2D scalar φ4 theory.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
from datetime import datetime
from tqdm import tqdm

from src.phi4_lattice import Phi4Lattice
from src.gan import LatticeGAN

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GAN on HMC samples')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing HMC samples')
    parser.add_argument('--samples_file', type=str, default=None,
                        help='Specific HMC samples file to use (if None, use latest)')
    parser.add_argument('--lattice_size', type=int, default=32,
                        help='Lattice size (L x L)')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of latent space')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Interval for saving models')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def load_samples(data_dir, samples_file=None):
    """
    Load HMC samples from file.
    
    Args:
        data_dir (str): Directory containing samples
        samples_file (str, optional): Specific file to load
        
    Returns:
        numpy.ndarray: Loaded samples
    """
    if samples_file is not None:
        # Load specific file
        samples_path = os.path.join(data_dir, samples_file)
        if not os.path.exists(samples_path):
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
    else:
        # Find the latest samples file
        sample_files = sorted(glob(os.path.join(data_dir, "hmc_samples_*.npy")))
        if not sample_files:
            raise FileNotFoundError(f"No HMC samples found in {data_dir}")
        samples_path = sample_files[-1]
    
    print(f"Loading samples from {samples_path}")
    samples = np.load(samples_path)
    print(f"Loaded {len(samples)} samples with shape {samples.shape}")
    
    return samples

def plot_gan_training(history, output_dir='results'):
    """
    Plot GAN training history.
    
    Args:
        history (dict): Training history
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gan_losses.png'))
    plt.close()
    
    # Plot discriminator scores
    plt.figure(figsize=(10, 6))
    plt.plot(history['d_real'], label='D(x) - Real')
    plt.plot(history['d_fake'], label='D(G(z)) - Fake')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Discriminator Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gan_scores.png'))
    plt.close()

def main():
    """Train the GAN on HMC samples."""
    args = parse_args()
    
    print(f"Starting GAN training with the following parameters:")
    print(f"  Lattice size: {args.lattice_size}x{args.lattice_size}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save interval: {args.save_interval}")
    
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)  # 确保 models 目录存在
    
    # Load HMC samples
    samples = load_samples(args.data_dir, args.samples_file)
    
    # Check lattice size
    L = samples.shape[1]
    if L != args.lattice_size:
        print(f"Warning: Loaded samples have lattice size {L}, but --lattice_size={args.lattice_size}")
        print(f"Using lattice size {L} from samples")
    
    # Initialize GAN
    print("Initializing GAN model...")
    gan = LatticeGAN(L, args.latent_dim, args.device)
    
    # Train GAN
    print(f"Starting GAN training for {args.epochs} epochs...")
    history = gan.train(
        samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval
    )
    
    # Save final model
    print("Training completed. Saving final model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(args.output_dir, f"gan_final_{timestamp}.pth")
    gan.save_model(final_model_path)
    
    # Create a symlink to the latest model
    latest_link = os.path.join(args.output_dir, "gan_latest.pth")
    if os.path.exists(latest_link) or os.path.islink(latest_link):
        os.remove(latest_link)
    
    # 使用绝对路径创建符号链接，避免相对路径问题
    abs_final_path = os.path.abspath(final_model_path)
    try:
        os.symlink(abs_final_path, latest_link)
        print(f"Created symlink from {latest_link} to {abs_final_path}")
    except Exception as e:
        print(f"Warning: Could not create symlink: {e}")
        print(f"Copying the model file instead...")
        import shutil
        shutil.copy2(final_model_path, latest_link)
    
    print(f"Final model saved to {final_model_path}")
    print(f"Latest model reference: {latest_link}")
    
    # Plot training history
    print("Plotting training history...")
    plot_gan_training(history)
    
    # Generate and visualize some samples
    print("Generating samples from trained GAN...")
    gan_samples = gan.generate_samples(16)
    
    # Plot samples
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(gan_samples[i], cmap='viridis')
        plt.axis('off')
    
    plt.suptitle('GAN-generated Samples')
    plt.tight_layout()
    plt.savefig('results/gan_samples.png')
    print("Sample visualization saved to 'results/gan_samples.png'")
    plt.close()
    
    print("GAN training and evaluation completed successfully!")
    return gan

if __name__ == '__main__':
    main() 