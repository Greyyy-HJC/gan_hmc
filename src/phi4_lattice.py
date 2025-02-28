import numpy as np

class Phi4Lattice:
    """
    Implementation of 2D scalar φ4 theory on a lattice with periodic boundary conditions.
    
    The action is given by:
    S = sum_x [ -2κ * sum_μ φ(x)φ(x+μ̂) + (1-2λ)φ(x)^2 + λφ(x)^4 ]
    
    where:
    - κ is the hopping parameter
    - λ is the quartic coupling
    """
    
    def __init__(self, L, kappa, lamb):
        """
        Initialize the lattice.
        
        Args:
            L (int): Lattice size (L x L)
            kappa (float): Hopping parameter
            lamb (float): Quartic coupling
        """
        self.L = L
        self.kappa = kappa
        self.lamb = lamb
        self.volume = L * L
        
    def random_config(self):
        """Generate a random lattice configuration."""
        return np.random.normal(0, 1, (self.L, self.L))
    
    def action(self, phi):
        """
        Calculate the action for a given field configuration.
        
        Args:
            phi (numpy.ndarray): Field configuration of shape (L, L)
            
        Returns:
            float: The action value
        """
        # Kinetic term (nearest-neighbor interaction)
        kinetic = 0.0
        for mu in range(2):  # 2D lattice
            # Shift in the mu direction with periodic boundary conditions
            if mu == 0:
                shifted = np.roll(phi, 1, axis=0)
            else:
                shifted = np.roll(phi, 1, axis=1)
            
            kinetic += np.sum(phi * shifted)
        
        # Potential terms
        quadratic = np.sum(phi**2)
        quartic = np.sum(phi**4)
        
        # Full action
        action = -2 * self.kappa * kinetic + (1 - 2 * self.lamb) * quadratic + self.lamb * quartic
        
        return action
    
    def force(self, phi):
        """
        Calculate the force (negative gradient of action) for HMC.
        
        Args:
            phi (numpy.ndarray): Field configuration of shape (L, L)
            
        Returns:
            numpy.ndarray: Force field of shape (L, L)
        """
        # Kinetic term contribution
        force = np.zeros_like(phi)
        for mu in range(2):
            if mu == 0:
                force += np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0)
            else:
                force += np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1)
        
        force *= 2 * self.kappa
        
        # Potential term contribution
        force -= 2 * (1 - 2 * self.lamb) * phi
        force -= 4 * self.lamb * phi**3
        
        return force
    
    def magnetization(self, phi):
        """
        Calculate the magnetization (order parameter).
        
        Args:
            phi (numpy.ndarray): Field configuration of shape (L, L)
            
        Returns:
            float: Magnetization value
        """
        return np.sum(phi) / self.volume
    
    def energy(self, phi):
        """
        Calculate the energy density.
        
        Args:
            phi (numpy.ndarray): Field configuration of shape (L, L)
            
        Returns:
            float: Energy density
        """
        return self.action(phi) / self.volume
    
    def susceptibility(self, phi):
        """
        Calculate the magnetic susceptibility.
        
        Args:
            phi (numpy.ndarray): Field configuration of shape (L, L)
            
        Returns:
            float: Susceptibility value
        """
        m = self.magnetization(phi)
        m2 = np.sum(phi**2) / self.volume
        return self.volume * (m2 - m**2)
    
    def binder_cumulant(self, phi):
        """
        Calculate the Binder cumulant.
        
        Args:
            phi (numpy.ndarray): Field configuration of shape (L, L)
            
        Returns:
            float: Binder cumulant value
        """
        m = self.magnetization(phi)
        m2 = np.mean(phi**2)
        m4 = np.mean(phi**4)
        return 1 - (m4 / (3 * m2**2)) 