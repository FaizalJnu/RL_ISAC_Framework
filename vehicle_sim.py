import numpy as np
from dataclasses import dataclass
from typing import Tuple, NamedTuple, Optional

class Angles(NamedTuple):
    azimuth: float
    elevation_angle: float
    elevation_azimuth: float
    aoa_in: float
    aoa_out: float

class ChannelAngles(NamedTuple):
    bs_to_target: Angles
    bs_to_ris: Angles
    ris_to_target: Angles

class Delays(NamedTuple):
    line_of_sight: float
    non_line_of_sight: float

class WirelessChannelSimulation:
    def __init__(self):
        # System Parameters
        self.Nb = 4  # Number of BS antennas (ULA)
        self.Nt = 4  # Number of Vehicle antennas
        self.Nx = 8  # Number of RIS elements in x-direction
        self.Ny = 8  # Number of RIS elements in y-direction
        self.Nr = self.Nx * self.Ny  # Total number of RIS elements
        self.Mb = 64  # Number of subcarriers
        self.N = 2048  # FFT size
        self.B = 20e6  # Bandwidth (20 MHz)
        self.fc = 28e9  # Carrier frequency
        self.c = 3e8  # Speed of light
        self.lambda_ = self.c / self.fc
        self.d = 0.5 * self.lambda_  # Antenna spacing
        
        # Environment Parameters
        self.x_size = 1000  # Environment size in meters
        self.y_size = 1000
        self.z_size = 100
        
        # Channel Parameters
        self.K_dB = 4  # Rician K-factor in dB
        self.K = 10**(self.K_dB/10)
        self.sigma = np.sqrt(1/(2*(self.K+1)))
        self.mu = np.sqrt(self.K/(self.K+1))
        
        # Initialize positions
        self.vehicle_pos = np.array([500, 500, 0])
        self.goal_pos = np.array([1000, 500, 0])
        self.bs_loc = np.array([900, 100, 20])
        self.ris_loc = np.array([200, 300, 40])
        
        # Initialize array positions
        self.bs_array_positions = self._init_bs_array()
        self.vehicle_array_positions = np.zeros((self.Nt, 3))
        self.ris_array_positions = self._init_ris_array()

    def _init_bs_array(self) -> np.ndarray:
        """Initialize BS array positions."""
        positions = np.zeros((self.Nb, 3))
        for i in range(self.Nb):
            positions[i] = self.bs_loc + np.array([0, i * self.d, 0])
        return positions

    def _init_ris_array(self) -> np.ndarray:
        """Initialize RIS array positions."""
        positions = np.zeros((self.Nr, 3))
        idx = 0
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                positions[idx] = self.ris_loc + np.array([ix * self.d, iy * self.d, 0])
                idx += 1
        return positions

    def generate_steering_vector(self, Nant: int, psi: float) -> np.ndarray:
        """Generate steering vector for ULA."""
        k = 2 * np.pi / self.lambda_
        n = np.arange(Nant)
        phase_terms = np.exp(1j * k * self.d * n * np.sin(psi))
        return phase_terms / np.sqrt(Nant)

    def generate_2d_steering_vector(self, Nx: int, phi_a: float, phi_e: float) -> np.ndarray:
        """Generate 2D steering vector for UPA."""
        N2 = Nx * Nx
        a_phi = np.zeros(N2, dtype=complex)
        k = 2 * np.pi / self.lambda_
        
        idx = 0
        for m in range(Nx):
            for n in range(Nx):
                phase_term = np.exp(1j * k * self.d * 
                                  (m * np.sin(phi_a) * np.sin(phi_e) + 
                                   n * np.cos(phi_e)))
                a_phi[idx] = phase_term
                idx += 1
        
        return a_phi / np.sqrt(N2)

    def generate_channels(self, angles: ChannelAngles) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate all channels (BS-Target, BS-RIS, RIS-Target)."""
        # Generate BS-Target channel
        H_bt = self._generate_H_bt(angles.bs_to_target)
        
        # Generate BS-RIS and RIS-Target channels
        H_br = self._generate_H_br(angles.bs_to_ris)
        H_rt = self._generate_H_rt(angles.ris_to_target)
        
        return H_bt, H_br, H_rt

    def _generate_H_bt(self, angles: Angles) -> np.ndarray:
        """Generate BS-Target channel."""
        a_psi_bt = self.generate_steering_vector(self.Nr, angles.aoa_in)
        a_psi_tb = self.generate_steering_vector(self.Nt, angles.aoa_out)
        return np.outer(a_psi_tb, a_psi_bt)

    def _generate_H_br(self, angles: Angles) -> np.ndarray:
        """Generate BS-RIS channel."""
        a_psi_br = self.generate_steering_vector(self.Nb, angles.azimuth)
        a_phi_abr = self.generate_2d_steering_vector(
            int(np.sqrt(self.Nr)), angles.elevation_azimuth, angles.elevation_angle)
        return np.outer(a_phi_abr, a_psi_br)

    def _generate_H_rt(self, angles: Angles) -> np.ndarray:
        """Generate RIS-Target channel."""
        a_psi_rt = self.generate_steering_vector(self.Nt, angles.azimuth)
        a_phi_art = self.generate_2d_steering_vector(
            int(np.sqrt(self.Nr)), angles.elevation_azimuth, angles.elevation_angle)
        return np.outer(a_psi_rt, a_phi_art)

    def update_vehicle_position(self, new_position: np.ndarray) -> None:
        """Update vehicle position and recalculate related parameters."""
        self.vehicle_pos = new_position
        # Update vehicle array positions based on new position
        for i in range(self.Nt):
            self.vehicle_array_positions[i] = self.vehicle_pos + np.array([0, i * self.d, 0])
            
    def simulate_downlink_transmission(self, delays: Delays, W: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Simulate downlink transmission with both LoS and NLoS paths.
        
        Args:
            delays: Delays for LoS and NLoS paths
            W: Optional beamforming matrix
            
        Returns:
            Tuple of H_NLoS, H_LoS channels and gamma_c parameter
        """
        # Generate transmit data for each subcarrier
        x = (np.random.randn(self.Mb) + 1j * np.random.randn(self.Mb)) / np.sqrt(2)
        
        # Generate beamforming matrix if not provided
        if W is None:
            W = np.zeros((self.Nb, self.Mb), dtype=complex)
            for i in range(self.Mb):
                w = (np.random.randn(self.Nb) + 1j * np.random.randn(self.Nb)) / np.sqrt(2 * self.Nb)
                W[:, i] = w / np.linalg.norm(w)
        
        # Generate basic channels
        angles = self.compute_geometric_parameters()  # You need to implement this
        H_bt, H_br, H_rt = self.generate_channels(angles)
        
        # Generate LoS and NLoS channels
        H_Los = self._generate_H_Los(H_bt, delays.line_of_sight)
        H_NLos = self._generate_H_NLoS(H_rt, H_br, delays.non_line_of_sight)
        
        return H_NLos, H_Los, 90.0  # gamma_c is set to 90 as in the original code

    def _generate_H_Los(self, H_bt: np.ndarray, tau_l: float) -> np.ndarray:
        """Generate Line of Sight channel matrix."""
        # Generate small-scale fading
        hl = (self.sigma * complex(np.random.randn(), np.random.randn())) + self.mu
        
        # Path loss parameters
        rho_l = 3
        gamma_l = np.sqrt(self.Nb * self.Nt) / np.sqrt(rho_l)
        
        # Generate H_Los for all subcarriers
        H_Los = np.zeros((self.Nt, self.Nr, self.N), dtype=complex)
        
        for n in range(self.N):
            phase = np.exp(1j * 2 * np.pi * self.B * n * tau_l / self.N)
            H_Los[:, :, n] = gamma_l * hl * H_bt * phase
            
        return H_Los

    def _generate_H_NLoS(self, H_rt: np.ndarray, H_br: np.ndarray, tau_nl: float) -> np.ndarray:
        """Generate Non-Line of Sight channel matrix."""
        # Generate small-scale fading
        hnl = (self.sigma * complex(np.random.randn(), np.random.randn())) + self.mu
        
        # Path loss parameters
        rho_nl = 4
        gamma_nl = np.sqrt(self.Nb * self.Nr) / np.sqrt(rho_nl)
        
        # Generate RIS reflection parameters
        rho_r = 1
        theta = 2 * np.pi * np.random.rand(self.Nr)
        u = rho_r * np.exp(1j * theta)
        Phi = np.diag(u)
        
        # Generate H_NLoS for all subcarriers
        H_NLoS = np.zeros((self.Nt, self.Nb, self.N), dtype=complex)
        
        for n in range(self.N):
            phase = np.exp(1j * 2 * np.pi * self.B * n * tau_nl / self.N)
            H_NLoS[:, :, n] = gamma_nl * hnl * H_rt @ Phi @ H_br * phase
            
        return H_NLoS
    
    def compute_geometric_parameters():
        # Define locations
        target_loc = np.array([500, 500, 0])  # Vehicle starts at ground level
        bs_loc = np.array([900, 100, 20])     # Base station location
        ris_loc = np.array([200, 300, 40])    # RIS location
        
        # Extract coordinates
        xb, yb, zb = bs_loc
        xr, yr, zr = ris_loc
        xt, yt = target_loc[0], target_loc[1]
        
        # Speed of light
        c = 3e8
        
        # Calculate 3D Euclidean distances
        L1 = np.sqrt((xb - xr)**2 + (yb - yr)**2 + (zb - zr)**2)
        L2 = np.sqrt((xr - xt)**2 + (yr - yt)**2 + zr**2)
        L3 = np.sqrt((xb - xt)**2 + (yb - yt)**2 + zb**2)
        
        # Calculate 2D (X-Y plane) projections
        L_proj1 = np.sqrt((xb - xr)**2 + (yb - yr)**2)
        L_proj2 = np.sqrt((xr - xt)**2 + (yr - yt)**2)
        L_proj3 = np.sqrt((xb - xt)**2 + (yb - yt)**2)
        
        # Calculate signal delays
        delays = {
            'line_of_sight': L3 / c,
            'non_line_of_sight': (L1 + L2) / c
        }
        
        # Calculate angles
        angles = {
            'bs_to_ris': {
                'azimuth': np.arcsin((zb - zr) / L1),
                'elevation_azimuth': np.arcsin((xb - xr) / L_proj1),
                'elevation_angle': np.arccos((zb - zr) / L1)
            },
            'ris_to_target': {
                'aoa': np.arcsin(zr / L2),
                'azimuth': np.arccos((yr - yt) / L_proj2),
                'elevation_angle': np.arccos(zr / L2)
            },
            'bs_to_target': {
                'aoa_in': np.arccos(zb/L3),
                'aoa_out': np.arcsin(zb/L3)
            }
        }
        
        return L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles
    
    # TODO: Implement PEB generation function here
    # TODO: Implement state function to derive reward from PEB
    # TODO: Implement action function to act on that reward changing  
    
    
def simulate_ris_v2x():
    WirelessChannelSimulation