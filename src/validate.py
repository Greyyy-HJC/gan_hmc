import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

class LatticeValidator:
    """
    Validation tools for lattice simulations with GANs.
    Performs statistical tests to ensure ergodicity and correct distribution.
    """
    
    def __init__(self, lattice):
        """
        Initialize the validator.
        
        Args:
            lattice: The Phi4Lattice instance
        """
        self.lattice = lattice
    
    def compare_action_distributions(self, hmc_samples, gan_samples, bins=50, figsize=(10, 6)):
        """
        Compare the action distributions of HMC and GAN samples.
        
        Args:
            hmc_samples (numpy.ndarray): HMC-generated samples
            gan_samples (numpy.ndarray): GAN-generated samples
            bins (int): Number of histogram bins
            figsize (tuple): Figure size
            
        Returns:
            tuple: (ks_statistic, p_value)
        """
        # Calculate actions
        hmc_actions = np.array([self.lattice.action(phi) for phi in tqdm(hmc_samples, desc="Calculating HMC actions")])
        gan_actions = np.array([self.lattice.action(phi) for phi in tqdm(gan_samples, desc="Calculating GAN actions")])
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(hmc_actions, gan_actions)
        
        # Plot histograms
        plt.figure(figsize=figsize)
        plt.hist(hmc_actions, bins=bins, alpha=0.5, label='HMC')
        plt.hist(gan_actions, bins=bins, alpha=0.5, label='GAN')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title(f'Action Distribution Comparison\nKS test: stat={ks_stat:.4f}, p-value={p_value:.4f}')
        plt.legend()
        plt.savefig('results/action_distribution.png')
        plt.close()
        
        print(f"Action distribution KS test: stat={ks_stat:.4f}, p-value={p_value:.4f}")
        
        return ks_stat, p_value
    
    def compare_observable_distributions(self, hmc_samples, gan_samples, bins=50, figsize=(15, 10)):
        """
        Compare distributions of various observables.
        
        Args:
            hmc_samples (numpy.ndarray): HMC-generated samples
            gan_samples (numpy.ndarray): GAN-generated samples
            bins (int): Number of histogram bins
            figsize (tuple): Figure size
            
        Returns:
            dict: Dictionary of KS test results for each observable
        """
        # Calculate observables
        hmc_obs = {
            'magnetization': np.array([self.lattice.magnetization(phi) for phi in tqdm(hmc_samples, desc="HMC magnetization")]),
            'energy': np.array([self.lattice.energy(phi) for phi in tqdm(hmc_samples, desc="HMC energy")]),
            'susceptibility': np.array([self.lattice.susceptibility(phi) for phi in tqdm(hmc_samples, desc="HMC susceptibility")]),
            'binder': np.array([self.lattice.binder_cumulant(phi) for phi in tqdm(hmc_samples, desc="HMC Binder cumulant")])
        }
        
        gan_obs = {
            'magnetization': np.array([self.lattice.magnetization(phi) for phi in tqdm(gan_samples, desc="GAN magnetization")]),
            'energy': np.array([self.lattice.energy(phi) for phi in tqdm(gan_samples, desc="GAN energy")]),
            'susceptibility': np.array([self.lattice.susceptibility(phi) for phi in tqdm(gan_samples, desc="GAN susceptibility")]),
            'binder': np.array([self.lattice.binder_cumulant(phi) for phi in tqdm(gan_samples, desc="GAN Binder cumulant")])
        }
        
        # Perform KS tests and plot histograms
        ks_results = {}
        
        plt.figure(figsize=figsize)
        
        for i, (name, hmc_values) in enumerate(hmc_obs.items()):
            gan_values = gan_obs[name]
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(hmc_values, gan_values)
            ks_results[name] = (ks_stat, p_value)
            
            # Plot
            plt.subplot(2, 2, i+1)
            plt.hist(hmc_values, bins=bins, alpha=0.5, label='HMC')
            plt.hist(gan_values, bins=bins, alpha=0.5, label='GAN')
            plt.xlabel(name.capitalize())
            plt.ylabel('Frequency')
            plt.title(f'{name.capitalize()} Distribution\nKS test: stat={ks_stat:.4f}, p-value={p_value:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/observable_distributions.png')
        plt.close()
        
        # Print results
        print("\nObservable Distribution KS Tests:")
        for name, (ks_stat, p_value) in ks_results.items():
            print(f"{name.capitalize()}: stat={ks_stat:.4f}, p-value={p_value:.4f}")
        
        return ks_results
    
    def calculate_autocorrelation(self, samples, observable='magnetization', max_lag=None):
        """
        Calculate the autocorrelation function for a given observable.
        
        Args:
            samples (numpy.ndarray): Lattice configurations
            observable (str): Observable to calculate autocorrelation for
            max_lag (int, optional): Maximum lag to calculate
            
        Returns:
            tuple: (lags, autocorr)
        """
        # Calculate the observable time series
        if observable == 'magnetization':
            obs_series = np.array([self.lattice.magnetization(phi) for phi in samples])
        elif observable == 'energy':
            obs_series = np.array([self.lattice.energy(phi) for phi in samples])
        elif observable == 'susceptibility':
            obs_series = np.array([self.lattice.susceptibility(phi) for phi in samples])
        elif observable == 'binder':
            obs_series = np.array([self.lattice.binder_cumulant(phi) for phi in samples])
        else:
            raise ValueError(f"Unknown observable: {observable}")
        
        # Center the series
        obs_series = obs_series - np.mean(obs_series)
        
        # Set maximum lag if not provided
        if max_lag is None:
            max_lag = len(obs_series) // 4
        
        # Calculate autocorrelation
        n = len(obs_series)
        variance = np.var(obs_series)
        
        if variance == 0:
            return np.arange(max_lag), np.zeros(max_lag)
        
        autocorr = np.zeros(max_lag)
        
        for lag in range(max_lag):
            # Calculate autocorrelation at lag
            c = np.sum(obs_series[:(n-lag)] * obs_series[lag:]) / ((n - lag) * variance)
            autocorr[lag] = c
        
        return np.arange(max_lag), autocorr
    
    def compare_autocorrelations(self, hmc_samples, gan_samples, observable='magnetization', max_lag=None, figsize=(10, 6)):
        """
        Compare autocorrelation functions between HMC and GAN+HMC samples.
        
        Args:
            hmc_samples (numpy.ndarray): HMC-generated samples
            gan_samples (numpy.ndarray): GAN+HMC-generated samples
            observable (str): Observable to calculate autocorrelation for
            max_lag (int, optional): Maximum lag to calculate
            figsize (tuple): Figure size
            
        Returns:
            tuple: (hmc_integrated_autocorr, gan_integrated_autocorr)
        """
        # Calculate autocorrelations
        hmc_lags, hmc_autocorr = self.calculate_autocorrelation(hmc_samples, observable, max_lag)
        gan_lags, gan_autocorr = self.calculate_autocorrelation(gan_samples, observable, max_lag)
        
        # Calculate integrated autocorrelation time
        # Use the first zero crossing or a maximum cutoff
        def integrated_autocorr(autocorr):
            # Find first negative or zero value
            for i, val in enumerate(autocorr):
                if val <= 0:
                    cutoff = i
                    break
            else:
                cutoff = len(autocorr) // 4  # Fallback
            
            # Integrate up to cutoff
            return 0.5 + np.sum(autocorr[1:cutoff])
        
        hmc_integrated = integrated_autocorr(hmc_autocorr)
        gan_integrated = integrated_autocorr(gan_autocorr)
        
        # Plot autocorrelation functions
        plt.figure(figsize=figsize)
        plt.plot(hmc_lags, hmc_autocorr, label=f'HMC (τ_int = {hmc_integrated:.2f})')
        plt.plot(gan_lags, gan_autocorr, label=f'GAN+HMC (τ_int = {gan_integrated:.2f})')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Lag')
        plt.ylabel(f'Autocorrelation of {observable}')
        plt.title(f'Autocorrelation Function Comparison for {observable}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'results/autocorrelation_{observable}.png')
        plt.close()
        
        print(f"\nIntegrated autocorrelation time for {observable}:")
        print(f"HMC: {hmc_integrated:.2f}")
        print(f"GAN+HMC: {gan_integrated:.2f}")
        print(f"Speedup factor: {hmc_integrated / gan_integrated:.2f}x")
        
        return hmc_integrated, gan_integrated
    
    def validate_ergodicity(self, hmc_samples, gan_samples, n_bins=10, figsize=(15, 10)):
        """
        Validate ergodicity by comparing distributions across different regions of phase space.
        
        Args:
            hmc_samples (numpy.ndarray): HMC-generated samples
            gan_samples (numpy.ndarray): GAN-generated samples
            n_bins (int): Number of bins for histograms
            figsize (tuple): Figure size
            
        Returns:
            bool: Whether the GAN samples pass ergodicity tests
        """
        # Calculate magnetization for both sample sets
        hmc_mag = np.array([self.lattice.magnetization(phi) for phi in tqdm(hmc_samples, desc="HMC magnetization")])
        gan_mag = np.array([self.lattice.magnetization(phi) for phi in tqdm(gan_samples, desc="GAN magnetization")])
        
        # Calculate energy for both sample sets
        hmc_energy = np.array([self.lattice.energy(phi) for phi in tqdm(hmc_samples, desc="HMC energy")])
        gan_energy = np.array([self.lattice.energy(phi) for phi in tqdm(gan_samples, desc="GAN energy")])
        
        # Create 2D histograms
        plt.figure(figsize=figsize)
        
        # HMC histogram
        plt.subplot(1, 2, 1)
        plt.hist2d(hmc_mag, hmc_energy, bins=n_bins, cmap='Blues')
        plt.colorbar(label='Frequency')
        plt.xlabel('Magnetization')
        plt.ylabel('Energy')
        plt.title('HMC Samples')
        
        # GAN histogram
        plt.subplot(1, 2, 2)
        plt.hist2d(gan_mag, gan_energy, bins=n_bins, cmap='Reds')
        plt.colorbar(label='Frequency')
        plt.xlabel('Magnetization')
        plt.ylabel('Energy')
        plt.title('GAN Samples')
        
        plt.tight_layout()
        plt.savefig('results/ergodicity_test.png')
        plt.close()
        
        # Perform statistical test for similarity of 2D distributions
        # We'll use a chi-square test on the binned data
        
        # Create 2D histograms for statistical test
        hmc_hist, x_edges, y_edges = np.histogram2d(hmc_mag, hmc_energy, bins=n_bins)
        gan_hist, _, _ = np.histogram2d(gan_mag, gan_energy, bins=[x_edges, y_edges])
        
        # Normalize histograms
        hmc_hist = hmc_hist / np.sum(hmc_hist)
        gan_hist = gan_hist / np.sum(gan_hist)
        
        # Calculate chi-square statistic
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        chi2_stat = np.sum((hmc_hist - gan_hist)**2 / (hmc_hist + gan_hist + epsilon))
        
        # Degrees of freedom: (n_bins - 1) * (n_bins - 1)
        dof = (n_bins - 1) * (n_bins - 1)
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        print(f"\nErgodicity test (2D distribution comparison):")
        print(f"Chi-square statistic: {chi2_stat:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"P-value: {p_value:.4f}")
        
        # Consider the test passed if p-value > 0.05
        passed = p_value > 0.05
        print(f"Ergodicity test {'passed' if passed else 'failed'}")
        
        return passed 