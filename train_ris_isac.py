import matlab.engine
import numpy as np
from ddpg import FLDDPG  # Import our DDPG implementation
import torch

class RISISACTrainer:
    def __init__(self):
        # Start MATLAB engine
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        
        # Initialize MATLAB simulation
        self.sim = self.eng.RISISAC_V2X_Sim()
        
        # Get state and action dimensions
        initial_state = self.eng.getState(self.sim)
        initial_state = np.array(initial_state).flatten()
        state_dim = len(initial_state)  # MATLAB returns 2D array
        # Action space: RIS phases (64) + throttle (1) + steering (1)
        action_dim = 64 + 2 
        print(f"Initializing FLDDPG with state_dim={state_dim}, action_dim={action_dim}")
        
        # Initialize DDPG agent
        self.agent = FLDDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            buffer_size=1000000,
            batch_size=64,
            gamma=0.99,
            tau=0.001,
            actor_lr=1e-4,
            critic_lr=1e-3
        )
        
    def calculate_metrics(precoder, sim):
        precoder_matlab = matlab.double(precoder.tolist())
        rate, peb = sim.calculatePerformanceMetrics(precoder_matlab, nargout=2)
        return float(rate), float(peb)
    
    @staticmethod
    def create_simple_precoder(self, Nb):
        """Create a simple uniform precoder matrix"""
        return np.eye(round(Nb)) / np.sqrt(Nb)
    
    def process_state(self, matlab_state):
        """Convert MATLAB state to proper numpy array format"""
        return np.array(matlab_state).flatten()
        
    def train(self, num_episodes=1000, max_steps=200):
        best_reward = float('-inf')
        episode_rewards = []
        self.peb_values = []
        self.episode_numbers = []
        peb_history = []
        print("Starting training...")
        Nb = self.eng.get_Nb(self.sim)
        for episode in range(num_episodes):
            # Reset environment
            # state = state.flatten() if len(state.shape) > 1 else state
            matlab_state = self.eng.reset(self.sim)
            state = np.array(self.eng.reset(self.sim))
            print(f"Initial state shape: {state.shape}")
            episode_reward = 0
            precoder = self.create_simple_precoder(self, Nb)
            
            current_peb = self.eng.calculatePerformanceMetrics(self.sim, precoder)
            self.peb_values.append(current_peb)
            self.episode_numbers.append(episode+1)
            
            for step in range(max_steps):
                # Select action using DDPG
                action = self.agent.select_action(state)
                
                # Convert numpy array to MATLAB array
                matlab_action = matlab.double(action.tolist())
                
                # Step the simulation
                next_matlab_state, reward, done = self.eng.step(self.sim, matlab_action, nargout=3)
                
                # Convert MATLAB outputs to numpy arrays
                next_state = self.process_state(next_matlab_state)
                reward = float(abs(reward))
                done = bool(done)
                
                # Store transition in replay buffer
                self.agent.replay_buffer.push(state, action, reward, next_state)
                
                # Update the networks
                self.agent.update()
                
                episode_reward += reward
                state = next_state
                
                # Log training progress
                if (step + 1) % 10 == 0:
                    print(f"Episode {episode + 1}, Step {step + 1}, Reward: {reward:.3f}")
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])  # Moving average of last 100 episodes
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Total reward: {episode_reward:.3f}")
            print(f"Average reward (last 100): {avg_reward:.3f}")
            print("----------------------------------------")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'actor_state_dict': self.agent.actor.state_dict(),
                    'critic_state_dict': self.agent.critic.state_dict(),
                    'best_reward': best_reward
                }, 'best_model.pth')
        
        return episode_rewards
    
    def test(self, num_episodes=10):
        """Test the trained agent"""
        print("Testing trained agent...")
        for episode in range(num_episodes):
            state = np.array(self.eng.reset(self.sim))
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 200:
                action = self.agent.select_action(state, explore=False)
                matlab_action = matlab.double(action.tolist())
                next_state, reward, done = self.eng.step(self.sim, matlab_action, nargout=3)
                
                state = np.array(next_state)
                episode_reward += float(abs(reward))
                step += 1
            
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.3f}")
    
    def close(self):
        """Clean up MATLAB engine"""
        self.eng.quit()

if __name__ == "__main__":
    # Create trainer instance
    trainer = RISISACTrainer()
    
    try:
        # Train the agent
        print("Starting training process...")
        rewards = trainer.train(num_episodes=100000, max_steps=10000)
        # Test the trained agent
        print("\nTesting trained agent...")
        trainer.test(num_episodes=10)
        
        # Plot training rewards
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_rewards.png')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.episode_numbers, trainer.peb_values)
        plt.title('Performance Error Bound (PEB) over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('PEB Value')
        plt.grid(True)
        plt.savefig('peb_values.png')
        plt.show()
        
    finally:
        # Clean up
        trainer.close()