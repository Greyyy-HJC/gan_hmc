import numpy as np
from tqdm import tqdm

class HMC:
    """
    Hybrid Monte Carlo (HMC) implementation for sampling from the φ4 theory.
    """
    
    def __init__(self, lattice, n_steps=10, step_size=0.1):
        """
        Initialize the HMC sampler.
        
        Args:
            lattice: The Phi4Lattice instance
            n_steps (int): Number of leapfrog steps
            step_size (float): Leapfrog step size
        """
        self.lattice = lattice
        self.n_steps = n_steps
        self.step_size = step_size
        self.acceptance_rate = 0.0
        self.n_accepted = 0
        self.n_total = 0
    
    def leapfrog_step(self, phi, pi):
        """
        Perform a single leapfrog integration step.
        
        Args:
            phi (numpy.ndarray): Field configuration
            pi (numpy.ndarray): Conjugate momentum
            
        Returns:
            tuple: Updated (phi, pi)
        """
        # Half-step update for momentum
        pi += 0.5 * self.step_size * self.lattice.force(phi)
        
        # Full-step update for field
        phi += self.step_size * pi
        
        # Half-step update for momentum
        pi += 0.5 * self.step_size * self.lattice.force(phi)
        
        return phi, pi
    
    def hamiltonian(self, phi, pi):
        """
        Calculate the Hamiltonian (energy function) for HMC.
        
        Args:
            phi (numpy.ndarray): Field configuration
            pi (numpy.ndarray): Conjugate momentum
            
        Returns:
            float: Hamiltonian value
        """
        # Kinetic energy term
        kinetic = 0.5 * np.sum(pi**2)
        
        # Potential energy term (action)
        potential = self.lattice.action(phi)
        
        return kinetic + potential
    
    def sample(self, phi_current):
        """
        Generate a new sample using HMC.
        
        Args:
            phi_current (numpy.ndarray): Current field configuration
            
        Returns:
            numpy.ndarray: New field configuration
        """
        # Copy the current configuration
        phi = np.copy(phi_current)
        
        # Sample random momentum
        pi = np.random.normal(0, 1, size=phi.shape)
        
        # Calculate initial Hamiltonian
        H_initial = self.hamiltonian(phi, pi)
        
        # Perform leapfrog integration
        for _ in range(self.n_steps):
            phi, pi = self.leapfrog_step(phi, pi)
        
        # Calculate final Hamiltonian
        H_final = self.hamiltonian(phi, pi)
        
        # Metropolis acceptance step
        delta_H = H_final - H_initial
        accept_prob = min(1.0, np.exp(-delta_H))
        
        # Accept or reject
        if np.random.random() < accept_prob:
            self.n_accepted += 1
            phi_new = phi
        else:
            phi_new = phi_current
        
        self.n_total += 1
        self.acceptance_rate = self.n_accepted / self.n_total
        
        return phi_new, accept_prob
    
    def run_chain(self, n_samples, n_burnin=1000, thin=1, initial_config=None):
        """
        Run the HMC chain to generate samples.
        
        Args:
            n_samples (int): Number of samples to generate
            n_burnin (int): Number of burn-in steps
            thin (int): Thinning factor
            initial_config (numpy.ndarray, optional): Initial configuration
            
        Returns:
            tuple: (samples, observables)
                - samples: Array of field configurations
                - observables: Dictionary of measured observables
        """
        L = self.lattice.L
        
        # Initialize configuration if not provided
        if initial_config is None:
            phi = self.lattice.random_config()
        else:
            phi = np.copy(initial_config)
        
        # Reset counters
        self.n_accepted = 0
        self.n_total = 0
        
        # Burn-in phase
        print("Burn-in phase...")
        for _ in tqdm(range(n_burnin)):
            phi, _ = self.sample(phi)
        
        # Sampling phase
        print(f"Sampling phase ({n_samples} samples)...")
        samples = np.zeros((n_samples, L, L))
        
        # Observables to track
        magnetization = np.zeros(n_samples)
        energy = np.zeros(n_samples)
        susceptibility = np.zeros(n_samples)
        binder = np.zeros(n_samples)
        
        sample_idx = 0
        for i in tqdm(range(n_samples * thin)):
            phi, _ = self.sample(phi)
            
            # Store sample and observables if not thinning
            if i % thin == 0:
                samples[sample_idx] = phi
                magnetization[sample_idx] = self.lattice.magnetization(phi)
                energy[sample_idx] = self.lattice.energy(phi)
                susceptibility[sample_idx] = self.lattice.susceptibility(phi)
                binder[sample_idx] = self.lattice.binder_cumulant(phi)
                sample_idx += 1
        
        # Collect observables
        observables = {
            'magnetization': magnetization,
            'energy': energy,
            'susceptibility': susceptibility,
            'binder_cumulant': binder,
            'acceptance_rate': self.acceptance_rate
        }
        
        print(f"HMC acceptance rate: {self.acceptance_rate:.4f}")
        
        return samples, observables
    
    def tune_parameters(self, target_acceptance=0.7, n_tuning_samples=1000, max_attempts=10):
        """
        Tune the HMC parameters to achieve a target acceptance rate.
        
        Args:
            target_acceptance (float): Target acceptance rate
            n_tuning_samples (int): Number of samples for tuning
            max_attempts (int): Maximum number of tuning attempts
            
        Returns:
            tuple: (n_steps, step_size)
        """
        print("Tuning HMC parameters...")
        
        phi = self.lattice.random_config()
        
        for attempt in range(max_attempts):
            # Reset counters
            self.n_accepted = 0
            self.n_total = 0
            
            # Run some samples
            for _ in tqdm(range(n_tuning_samples)):
                phi, _ = self.sample(phi)
            
            print(f"Attempt {attempt+1}: n_steps={self.n_steps}, step_size={self.step_size:.6f}, "
                  f"acceptance_rate={self.acceptance_rate:.4f}")
            
            # Adjust step size based on acceptance rate
            if abs(self.acceptance_rate - target_acceptance) < 0.05:
                print("Tuning successful!")
                break
            
            if self.acceptance_rate < target_acceptance:
                self.step_size *= 0.8  # Decrease step size
            else:
                self.step_size *= 1.2  # Increase step size
        
        return self.n_steps, self.step_size 