import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

class Generator(nn.Module):
    """
    Generator network for the GAN.
    Maps from latent space to lattice configurations.
    """
    
    def __init__(self, latent_dim, lattice_size):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.lattice_size = lattice_size
        self.output_dim = lattice_size * lattice_size
        
        # Define the network architecture
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            
            nn.Linear(2048, self.output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Generated lattice configurations of shape (batch_size, L*L)
        """
        return self.model(z)
    
    def generate(self, z=None, batch_size=1, device='cpu'):
        """
        Generate lattice configurations.
        
        Args:
            z (torch.Tensor, optional): Latent vectors
            batch_size (int): Number of configurations to generate
            device (str): Device to use
            
        Returns:
            torch.Tensor: Generated lattice configurations of shape (batch_size, L, L)
        """
        # Ensure we're in eval mode
        self.eval()
        
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
        
        with torch.no_grad():
            # Handle the case of a single sample for BatchNorm
            if z.size(0) == 1 and not self.training:
                # For a single sample, we need to temporarily switch to eval mode
                # to avoid BatchNorm issues
                generated = self.forward(z)
            else:
                generated = self.forward(z)
                
            # Reshape to lattice configuration
            return generated.view(z.size(0), self.lattice_size, self.lattice_size)


class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.
    Classifies lattice configurations as real or fake.
    """
    
    def __init__(self, lattice_size):
        super(Discriminator, self).__init__()
        
        self.lattice_size = lattice_size
        self.input_dim = lattice_size * lattice_size
        
        # Define the network architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x (torch.Tensor): Lattice configurations of shape (batch_size, L, L)
            
        Returns:
            torch.Tensor: Probability that input is real, shape (batch_size, 1)
        """
        # Flatten the input
        x_flat = x.view(x.size(0), -1)
        return self.model(x_flat)


class LatticeGAN:
    """
    GAN for generating lattice configurations.
    """
    
    def __init__(self, lattice_size, latent_dim=256, device=None):
        """
        Initialize the GAN.
        
        Args:
            lattice_size (int): Size of the lattice (L)
            latent_dim (int): Dimension of the latent space
            device (str, optional): Device to use ('cuda' or 'cpu')
        """
        self.lattice_size = lattice_size
        self.latent_dim = latent_dim
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.generator = Generator(latent_dim, lattice_size).to(self.device)
        self.discriminator = Discriminator(lattice_size).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real': [],
            'd_fake': []
        }
    
    def train(self, samples, epochs=5000, batch_size=64, save_interval=500):
        """
        Train the GAN.
        
        Args:
            samples (numpy.ndarray): Training samples of shape (n_samples, L, L)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            save_interval (int): Interval for saving models
            
        Returns:
            dict: Training history
        """
        # Convert samples to PyTorch tensors
        samples_tensor = torch.tensor(samples, dtype=torch.float32)
        dataset = TensorDataset(samples_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Labels for real and fake samples
        real_label = 1.0
        fake_label = 0.0
        
        print(f"Starting GAN training for {epochs} epochs...")
        
        # Use tqdm for the epoch loop
        for epoch in tqdm(range(epochs), desc="Training GAN", unit="epoch"):
            g_losses = []
            d_losses = []
            d_real_scores = []
            d_fake_scores = []
            
            # Use tqdm for the batch loop if there are many batches
            n_batches = len(dataloader)
            batch_iterator = dataloader
            if n_batches > 10:  # Only use tqdm for batch loop if there are many batches
                batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                                     leave=False, unit="batch")
            
            for batch_idx, (real_samples,) in enumerate(batch_iterator):
                batch_size = real_samples.size(0)
                real_samples = real_samples.to(self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                
                # Real samples
                real_labels = torch.full((batch_size, 1), real_label, device=self.device)
                real_output = self.discriminator(real_samples)
                d_real_loss = self.criterion(real_output, real_labels)
                d_real_loss.backward()
                d_real_score = real_output.mean().item()
                
                # Fake samples
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_samples = self.generator(z)
                fake_labels = torch.full((batch_size, 1), fake_label, device=self.device)
                fake_output = self.discriminator(fake_samples.detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)
                d_fake_loss.backward()
                d_fake_score = fake_output.mean().item()
                
                # Combined loss
                d_loss = d_real_loss + d_fake_loss
                self.d_optimizer.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                self.g_optimizer.zero_grad()
                
                # Try to fool the discriminator
                fake_output = self.discriminator(fake_samples)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                d_real_scores.append(d_real_score)
                d_fake_scores.append(d_fake_score)
            
            # Calculate average losses for the epoch
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            avg_d_real = np.mean(d_real_scores)
            avg_d_fake = np.mean(d_fake_scores)
            
            # Update history
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['d_real'].append(avg_d_real)
            self.history['d_fake'].append(avg_d_fake)
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                tqdm.write(f"Epoch [{epoch+1}/{epochs}] | "
                      f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | "
                      f"D(x): {avg_d_real:.4f} | D(G(z)): {avg_d_fake:.4f}")
            
            # Save model
            if (epoch + 1) % save_interval == 0:
                self.save_model(f"models/gan_epoch_{epoch+1}.pth")
                tqdm.write(f"Model saved at epoch {epoch+1}")
        
        # Save final model
        self.save_model("models/gan_final.pth")
        
        print("GAN training completed!")
        return self.history
    
    def generate_samples(self, n_samples, reshape=True):
        """
        Generate lattice configurations using the trained generator.
        
        Args:
            n_samples (int): Number of samples to generate
            reshape (bool): Whether to reshape to (n_samples, L, L)
            
        Returns:
            numpy.ndarray: Generated samples
        """
        self.generator.eval()
        
        # Generate samples in batches to avoid memory issues
        batch_size = 100
        n_batches = int(np.ceil(n_samples / batch_size))
        samples = []
        
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Generating samples"):
                curr_batch_size = min(batch_size, n_samples - i * batch_size)
                z = torch.randn(curr_batch_size, self.latent_dim, device=self.device)
                batch_samples = self.generator(z)
                
                if reshape:
                    batch_samples = batch_samples.view(curr_batch_size, self.lattice_size, self.lattice_size)
                
                samples.append(batch_samples.cpu().numpy())
        
        # Concatenate batches
        samples = np.concatenate(samples, axis=0)
        return samples
    
    def save_model(self, path):
        """
        Save the GAN model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history,
            'lattice_size': self.lattice_size,
            'latent_dim': self.latent_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a saved GAN model.
        
        Args:
            path (str): Path to the saved model
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # 使用 weights_only=False 加载完整的模型信息，包括 latent_dim 和其他元数据
            # 这会产生 FutureWarning，但对于我们的用例是必要的
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Check if lattice size and latent dim match
            model_lattice_size = checkpoint.get('lattice_size')
            model_latent_dim = checkpoint.get('latent_dim')
            
            if model_lattice_size != self.lattice_size:
                print(f"Warning: Loaded model has different lattice size "
                      f"(L={model_lattice_size}) than current model (L={self.lattice_size})")
                
            if model_latent_dim != self.latent_dim:
                print(f"Warning: Loaded model has different latent dimension "
                      f"(latent_dim={model_latent_dim}) than current model (latent_dim={self.latent_dim})")
                print("This will cause parameter size mismatch. Reinitializing the model with the correct latent dimension.")
                
                # Reinitialize the model with the correct latent dimension
                self.latent_dim = model_latent_dim
                self.generator = Generator(model_latent_dim, self.lattice_size).to(self.device)
                self.discriminator = Discriminator(self.lattice_size).to(self.device)
                
                # Reinitialize optimizers
                self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            
            # Load model parameters
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.history = checkpoint['history']
            
            print(f"Model successfully loaded from {path}")
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {e}")
    
    def get_latent_vector(self, phi, n_steps=1000, lr=0.01):
        """
        Find a latent vector that generates a configuration close to the given one.
        
        Args:
            phi (numpy.ndarray): Target configuration of shape (L, L)
            n_steps (int): Number of optimization steps
            lr (float): Learning rate
            
        Returns:
            torch.Tensor: Optimized latent vector
        """
        self.generator.eval()
        
        # Convert target to tensor
        target = torch.tensor(phi, dtype=torch.float32).to(self.device)
        target = target.view(1, self.lattice_size, self.lattice_size)
        
        # Initialize latent vector
        z = torch.randn(1, self.latent_dim, requires_grad=True, device=self.device)
        optimizer = optim.Adam([z], lr=lr)
        
        # MSE loss
        mse_loss = nn.MSELoss()
        
        # Optimize latent vector
        for step in tqdm(range(n_steps), desc="Optimizing latent vector"):
            optimizer.zero_grad()
            
            # Generate configuration
            generated = self.generator.generate(z)
            
            # Calculate loss
            loss = mse_loss(generated, target)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                tqdm.write(f"Step [{step+1}/{n_steps}] | Loss: {loss.item():.6f}")
        
        return z.detach() 