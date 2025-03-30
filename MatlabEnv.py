import matlab.engine
import numpy as np
import gym
from gym import spaces

class MatlabEnv(gym.Env):
    """Custom Gym environment to interact with MATLAB-based RIS-ISAC-V2X simulation."""

    def __init__(self, Nr, H_size):
        super(MatlabEnv, self).__init__()

        self.eng = matlab.engine.start_matlab() 

        # Initialize MATLAB simulation
        self.sim = self.eng.RISISAC_V2X_Sim()
        
        self.Nr = Nr
        self.H_size = H_size

        # Define action space (RIS phases)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.Nr,), dtype=np.float32)

        # Define observation space
        state_size = 2 * self.Nr + 1 + 2 * self.H_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

    def reset(self):
        """Resets the MATLAB environment and returns the initial state."""
        state = self.eng.RIS_ISAC_V2XSim('reset')
        return np.array(state).flatten()

    def step(self, action):
        """Takes an action, returns next_state, reward, done, and additional info."""
        action_matlab = matlab.double(action.tolist())  # Convert action to MATLAB format
        next_state, reward, done = self.eng.RIS_ISAC_V2XSim('step', action_matlab, nargout=3)
        
        return np.array(next_state).flatten(), float(reward), bool(done), {}

    def close(self):
        """Closes MATLAB engine."""
        self.eng.quit()
