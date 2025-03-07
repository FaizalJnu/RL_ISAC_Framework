import matlab.engine
import numpy as np
from ddpg import FLDDPG  # Import our DDPG implementation
import torch
import matplotlib.pyplot as plt

class RISISACTrainer:
    def __init__(self):
        # Start MATLAB engine
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab() 

        # Initialize MATLAB simulation
        self.sim = self.eng.RISISAC_V2X_Sim()
        # print(self.eng.methods(self.sim))
        # print(type(self.sim))
        
        # Get state and action dimensions
        initial_state = self.eng.getState(self.sim)
        initial_state = np.array(initial_state).flatten()
        state_dim = len(initial_state)
        action_dim = 64  # RIS phases (64) + throttle (1) + steering (1)
        print(f"Initializing FLDDPG with state_dim={state_dim}, action_dim={action_dim}")
        
        # Initialize DDPG agent with improved parameters
        self.agent = FLDDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            buffer_size=1000000,
            batch_size=64,
            gamma=0.99,
            tau=0.001,
            actor_lr=1e-4,
            critic_lr=1e-3,
            lr_decay_rate=0.995,
            min_lr=1e-6
        )
        
        # Initialize metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'peb_values': [],
            'actor_losses': [],
            'critic_losses': [],
            'learning_rates': []
        }
        
        # Best metrics tracking
        self.best_metrics = {
            'reward': float('-inf'),
            'peb': float('inf'),
            'reward_episode': 0,
            'peb_episode': 0
        }

    @staticmethod
    def create_simple_precoder(self, Nb):
        """Create a simple uniform precoder matrix"""
        return np.eye(round(Nb)) / np.sqrt(Nb)
    
    def process_state(self, matlab_state):
        """Convert MATLAB state to proper numpy array format"""
        return np.array(matlab_state).flatten()
    
    def save_checkpoint(self, episode, metrics, checkpoint_type='best_peb'):
        """Save model checkpoint with detailed metrics"""
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, f'{checkpoint_type}_model.pth')
    
    def plot_training_progress(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot PEB values
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['peb_values'])
        plt.title('Position Error Bound (PEB)')
        plt.xlabel('Episode')
        plt.ylabel('PEB')
        plt.yscale('log')
        
        # Plot losses
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['actor_losses'], label='Actor')
        plt.plot(self.metrics['critic_losses'], label='Critic')
        plt.title('Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot learning rates
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def train(self, num_episodes, max_steps, target_peb):
        print("Starting training...")
        Nb = self.eng.get_Nb(self.sim)
        
        for episode in range(num_episodes):
            # Reset environment
            matlab_state = self.eng.reset(self.sim)
            state = self.process_state(matlab_state)
            # print("State dtype:", matlab_state.dtype)
            # print("Contains complex values?", np.iscomplexobj(matlab_state))
            episode_reward = 0
            episode_losses = {'actor': [], 'critic': []}
            
            # Initialize episode precoder
            # precoder = self.create_simple_precoder(self, Nb)
            step_counter = 0;
            
            for step in range(max_steps):
                step_counter = step_counter + 1
                # Select action with exploration
                action = self.agent.select_action(state, explore=True)
                
                # Convert and execute action
                matlab_action = matlab.double(action.tolist())
                
                next_matlab_state, reward, done = self.eng.step(self.sim, matlab_action, nargout=3)
                current_peb = self.eng.calculatePerformanceMetrics(self.sim)
                # print(f"Raw MATLAB Reward: {reward}, PEB: {current_peb}")
                
                # Process step results
                next_state = self.process_state(next_matlab_state)
                # reward = float(abs(reward))  # Using absolute reward
                done = bool(done)
                
                # Store transition and update networks
                self.agent.replay_buffer.push(state, action, reward, next_state)
                actor_loss, critic_loss = self.agent.update()
                
                # Track step metrics
                episode_reward += reward
                if actor_loss is not None:
                    episode_losses['actor'].append(actor_loss)
                    episode_losses['critic'].append(critic_loss)
                
                state = next_state
                
                if done:
                    break
            episode_reward = episode_reward/step_counter    
            # Update metrics
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['peb_values'].append(current_peb)
            if episode_losses['actor']:
                self.metrics['actor_losses'].append(np.mean(episode_losses['actor']))
                self.metrics['critic_losses'].append(np.mean(episode_losses['critic']))
            self.metrics['learning_rates'].append(self.agent.current_actor_lr)
            
            # Update best metrics and save checkpoints
            if episode_reward > self.best_metrics['reward']:
                self.best_metrics['reward'] = episode_reward
                self.best_metrics['reward_episode'] = episode
                self.save_checkpoint(episode, self.metrics, 'best_reward')
            
            if abs(current_peb) < abs(self.best_metrics['peb']):
                self.best_metrics['peb'] = current_peb
                self.best_metrics['peb_episode'] = episode
                self.save_checkpoint(episode, self.metrics, 'best_peb')
            
            # Decay learning rates
            decay_rate = self.agent.decay_learning_rates()
            

            # Print progress
            if episode % 10 == 0:
                self.plot_training_progress()
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Reward: {episode_reward:.3f}")
                print(f"PEB: {current_peb:.6f}")
                print(f"Best PEB: {self.best_metrics['peb']:.6f}")
                print(f"Learning Rate: {self.agent.current_actor_lr:.6f}")
                print(f"Buffer Size: {len(self.agent.replay_buffer)}")
                print(f"Decay rate is: {decay_rate}")
                print("-" * 50)
            
            # Early stopping check
            if episode > 1000 and np.mean(self.metrics['peb_values'][-1000:]) < target_peb:
                print(f"Early stopping at episode {episode} - Target PEB achieved")
                break
        
        return self.metrics
    
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
    print(type(trainer.sim))
    
    try:
        # Train the agent
        print("Starting training process...")
        # rewards = trainer.train(num_episodes=100000, max_steps=10000, target_peb=12)
        metrics = trainer.train(num_episodes=300, max_steps=10000, target_peb=0)

        # Extract episode rewards
        rewards = metrics['episode_rewards']
        episodes = list(range(len(rewards)))

        # Test the trained agent
        print("\nTesting trained agent...")
        trainer.test(num_episodes=10)
        
        # Plot training rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_rewards.png')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, metrics['peb_values'])
        plt.title('Performance Error Bound (PEB) over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('PEB Value')
        plt.grid(True)
        plt.savefig('peb_values.png')
        plt.show()
        
    finally:
        # Clean up
        trainer.close()