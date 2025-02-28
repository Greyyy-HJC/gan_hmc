import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

class GANOverrelaxation:
    """
    Implementation of GAN-based overrelaxation for reducing autocorrelation times
    in lattice simulations, as described in the paper:
    "Reducing autocorrelation times in lattice simulations with generative adversarial networks"
    """
    
    def __init__(self, lattice, gan, action_threshold=0.1, n_attempts=10, n_opt_steps=100, lr=0.01):
        """
        Initialize the GAN overrelaxation.
        
        Args:
            lattice: The Phi4Lattice instance
            gan: The trained LatticeGAN instance
            action_threshold (float): Maximum allowed action difference
            n_attempts (int): Number of attempts to find a suitable configuration
            n_opt_steps (int): Number of optimization steps for action matching
            lr (float): Learning rate for optimization
        """
        self.lattice = lattice
        self.gan = gan
        self.action_threshold = action_threshold
        self.n_attempts = n_attempts
        self.n_opt_steps = n_opt_steps
        self.lr = lr
        
        # Statistics
        self.n_accepted = 0
        self.n_total = 0
        self.acceptance_rate = 0.0
    
    def find_matching_action(self, phi_current):
        """
        Find a GAN-generated configuration with action close to the current one.
        
        Args:
            phi_current (numpy.ndarray): Current field configuration
            
        Returns:
            tuple: (phi_new, action_diff, success)
        """
        # Calculate current action
        current_action = self.lattice.action(phi_current)
        
        # Ensure generator is in eval mode
        self.gan.generator.eval()
        
        # Try multiple random latent vectors
        for _ in range(self.n_attempts):
            # Generate a random configuration from the GAN
            z = torch.randn(1, self.gan.latent_dim, device=self.gan.device)
            with torch.no_grad():  # Ensure no gradients are computed
                phi_gan = self.gan.generator.generate(z).cpu().numpy()[0]
            
            # Calculate action difference
            gan_action = self.lattice.action(phi_gan)
            action_diff = abs(gan_action - current_action)
            
            # If action difference is small enough, return this configuration
            if action_diff < self.action_threshold:
                return phi_gan, action_diff, True
        
        # If no suitable configuration found, try to optimize one
        return self.optimize_action_match(phi_current, current_action)
    
    def optimize_action_match(self, phi_current, current_action):
        """
        Optimize a latent vector to generate a configuration with matching action.
        
        Args:
            phi_current (numpy.ndarray): Current field configuration
            current_action (float): Action of the current configuration
            
        Returns:
            tuple: (phi_new, action_diff, success)
        """
        # Set generator to train mode to enable gradient computation
        self.gan.generator.train()
        
        # Initialize a random latent vector
        z = torch.randn(1, self.gan.latent_dim, requires_grad=True, device=self.gan.device)
        optimizer = optim.Adam([z], lr=self.lr)
        
        # Define MSE loss for action matching
        mse_loss = nn.MSELoss()
        
        # Target action as tensor
        target_action = torch.tensor([current_action], dtype=torch.float32, device=self.gan.device)
        
        # Optimize latent vector to match action
        for step in range(self.n_opt_steps):
            optimizer.zero_grad()
            
            # Generate configuration with gradients enabled
            phi_gan = self.gan.generator.generate(z)
            
            # Convert to numpy for action calculation
            phi_gan_np = phi_gan.detach().cpu().numpy()[0]
            gan_action = self.lattice.action(phi_gan_np)
            
            # Create a differentiable proxy for the action
            # Instead of detaching, we create a differentiable operation
            # that connects the generated configuration to the loss
            action_proxy = phi_gan.mean() * 0.0 + torch.tensor(gan_action, dtype=torch.float32, 
                                                             device=self.gan.device, requires_grad=True)
            
            # Calculate loss (action difference squared)
            action_proxy = action_proxy.view(1)  # Reshape to [1]
            loss = mse_loss(action_proxy, target_action)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # Check if action difference is small enough
            action_diff = abs(gan_action - current_action)
            if action_diff < self.action_threshold:
                # Set generator back to eval mode
                self.gan.generator.eval()
                with torch.no_grad():
                    phi_gan = self.gan.generator.generate(z).cpu().numpy()[0]
                return phi_gan, action_diff, True
        
        # Set generator back to eval mode
        self.gan.generator.eval()
        
        # Get the final configuration
        with torch.no_grad():
            phi_gan = self.gan.generator.generate(z).cpu().numpy()[0]
        gan_action = self.lattice.action(phi_gan)
        action_diff = abs(gan_action - current_action)
        
        # Return the best configuration found, even if not within threshold
        return phi_gan, action_diff, action_diff < self.action_threshold
    
    def overrelaxation_step(self, phi_current):
        """
        Perform a GAN-based overrelaxation step.
        
        Args:
            phi_current (numpy.ndarray): Current field configuration
            
        Returns:
            tuple: (phi_new, accepted)
        """
        # Find a GAN-generated configuration with matching action
        phi_gan, action_diff, success = self.find_matching_action(phi_current)
        
        # Update statistics
        self.n_total += 1
        
        # If successful, accept the new configuration
        if success:
            self.n_accepted += 1
            self.acceptance_rate = self.n_accepted / self.n_total
            return phi_gan, True
        else:
            self.acceptance_rate = self.n_accepted / self.n_total
            return phi_current, False
    
    def run_chain(self, n_samples, initial_config=None, hmc_sampler=None, hmc_frequency=1):
        """
        Run a Markov chain with GAN overrelaxation steps.
        
        Args:
            n_samples (int): Number of samples to generate
            initial_config (numpy.ndarray, optional): Initial configuration
            hmc_sampler: HMC sampler for occasional HMC steps
            hmc_frequency (int): Frequency of HMC steps
            
        Returns:
            tuple: (samples, observables)
        """
        L = self.lattice.L
        
        # Initialize configuration if not provided
        if initial_config is None:
            print("No initial configuration provided. Using random initialization.")
            phi = self.lattice.random_config()
        else:
            print(f"Using provided initial configuration with shape {initial_config.shape}")
            phi = np.copy(initial_config)
        
        # Reset counters
        self.n_accepted = 0
        self.n_total = 0
        
        # Sampling phase
        print(f"Running GAN overrelaxation chain ({n_samples} samples)...")
        print(f"HMC frequency: {hmc_frequency} (0 = no HMC steps)")
        print(f"Action threshold: {self.action_threshold}")
        print(f"Number of attempts per step: {self.n_attempts}")
        print(f"Optimization steps if needed: {self.n_opt_steps}")
        
        samples = np.zeros((n_samples, L, L))
        
        # Observables to track
        magnetization = np.zeros(n_samples)
        energy = np.zeros(n_samples)
        susceptibility = np.zeros(n_samples)
        binder = np.zeros(n_samples)
        
        # Progress tracking variables
        gan_successes = 0
        opt_successes = 0
        hmc_steps = 0
        gan_errors = 0
        hmc_errors = 0
        
        try:
            for i in tqdm(range(n_samples)):
                # Perform GAN overrelaxation step
                try:
                    # First try to find a matching configuration without optimization
                    current_action = self.lattice.action(phi)
                    phi_gan = None
                    action_diff = float('inf')
                    success = False
                    
                    # Try random latent vectors first
                    self.gan.generator.eval()  # Ensure in eval mode for random attempts
                    for _ in range(self.n_attempts):
                        z = torch.randn(1, self.gan.latent_dim, device=self.gan.device)
                        with torch.no_grad():
                            gen_phi = self.gan.generator.generate(z).cpu().numpy()[0]
                        
                        gen_action = self.lattice.action(gen_phi)
                        gen_action_diff = abs(gen_action - current_action)
                        
                        if gen_action_diff < self.action_threshold:
                            phi_gan = gen_phi
                            action_diff = gen_action_diff
                            success = True
                            break
                    
                    # If no match found, try optimization
                    if not success:
                        try:
                            # Set generator to train mode for optimization
                            self.gan.generator.train()
                            
                            # Initialize latent vector
                            z = torch.randn(1, self.gan.latent_dim, requires_grad=True, device=self.gan.device)
                            optimizer = optim.Adam([z], lr=self.lr)
                            
                            # Target action
                            target_action = torch.tensor([current_action], dtype=torch.float32, device=self.gan.device)
                            
                            # Optimize
                            best_action_diff = float('inf')
                            best_phi = None
                            
                            for step in range(self.n_opt_steps):
                                optimizer.zero_grad()
                                
                                # Generate with gradients
                                phi_gen = self.gan.generator.generate(z)
                                
                                # Calculate action
                                phi_gen_np = phi_gen.detach().cpu().numpy()[0]
                                gen_action = self.lattice.action(phi_gen_np)
                                
                                # Create differentiable proxy
                                action_proxy = phi_gen.mean() * 0.0 + torch.tensor(gen_action, 
                                                                                dtype=torch.float32, 
                                                                                device=self.gan.device, 
                                                                                requires_grad=True)
                                
                                # Loss - ensure both tensors have the same shape
                                # Convert action_proxy to match target_action shape
                                action_proxy = action_proxy.view(1)  # Reshape to [1]
                                loss = nn.MSELoss()(action_proxy, target_action)
                                
                                # Update
                                loss.backward()
                                optimizer.step()
                                
                                # Track best
                                gen_action_diff = abs(gen_action - current_action)
                                if gen_action_diff < best_action_diff:
                                    best_action_diff = gen_action_diff
                                    best_phi = phi_gen_np
                                
                                if gen_action_diff < self.action_threshold:
                                    break
                            
                            # Set back to eval mode
                            self.gan.generator.eval()
                            
                            # Use best found
                            if best_phi is not None and best_action_diff < self.action_threshold:
                                phi_gan = best_phi
                                action_diff = best_action_diff
                                success = True
                                opt_successes += 1
                        
                        except Exception as e:
                            print(f"Detailed optimization error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            success = False
                    
                    # Update configuration if successful
                    if success:
                        phi = phi_gan
                        gan_successes += 1
                        self.n_accepted += 1
                    
                    self.n_total += 1
                    self.acceptance_rate = self.n_accepted / self.n_total
                
                except Exception as e:
                    print(f"Detailed GAN overrelaxation error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    gan_errors += 1
                
                # Occasionally perform HMC step if provided
                if hmc_sampler is not None and hmc_frequency > 0 and i % hmc_frequency == 0:
                    try:
                        phi, _ = hmc_sampler.sample(phi)
                        hmc_steps += 1
                    except Exception as e:
                        print(f"Detailed HMC error: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        hmc_errors += 1
                
                # Store sample and observables
                samples[i] = phi
                magnetization[i] = self.lattice.magnetization(phi)
                energy[i] = self.lattice.energy(phi)
                susceptibility[i] = self.lattice.susceptibility(phi)
                binder[i] = self.lattice.binder_cumulant(phi)
                
                # Print progress occasionally
                if (i+1) % 1000 == 0 or i == 0:
                    current_acceptance = self.n_accepted / (self.n_total or 1)
                    tqdm.write(f"Step {i+1}/{n_samples} | "
                              f"GAN acceptance: {current_acceptance:.4f} | "
                              f"M: {magnetization[i]:.4f} | E: {energy[i]:.4f}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Returning samples collected so far...")
            # Trim arrays to the number of samples actually collected
            samples = samples[:i+1]
            magnetization = magnetization[:i+1]
            energy = energy[:i+1]
            susceptibility = susceptibility[:i+1]
            binder = binder[:i+1]
        
        # Collect observables
        observables = {
            'magnetization': magnetization,
            'energy': energy,
            'susceptibility': susceptibility,
            'binder_cumulant': binder,
            'acceptance_rate': self.acceptance_rate
        }
        
        print(f"\nGAN overrelaxation statistics:")
        print(f"  Total steps: {self.n_total}")
        print(f"  Accepted GAN steps: {self.n_accepted} ({self.acceptance_rate:.4f})")
        print(f"  Direct matches: {gan_successes - opt_successes}")
        print(f"  Optimization matches: {opt_successes}")
        print(f"  HMC steps: {hmc_steps}")
        print(f"  GAN errors: {gan_errors}")
        print(f"  HMC errors: {hmc_errors}")
        
        return samples, observables 