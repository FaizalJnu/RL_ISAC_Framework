import matlab.engine
import numpy as np
from flddpg import FLDDPG  # Import our DDPG implementation
import torch
import matplotlib.pyplot as plt
import os
import time
from drlisac import DDPGagent

class RISISACTrainer:
    def __init__(self):
        # Start MATLAB engine
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        
        # if torch.cuda.is_available():
        #     self.eng.eval("parpool('local', 4);", nargout=0)
        # else:
        #     self.eng.eval("parpool('local', 10);", nargout=0) 

        # Initialize MATLAB simulation
        self.sim = self.eng.RISISAC_V2X_Sim()
        # Get state and action dimensions
        initial_state = self.eng.getState(self.sim)
        initial_state = np.array(initial_state).flatten()
        state_dim = len(initial_state)
        action_dim = 64  # RIS phases (64)
        print(f"Initializing FLDDPG with state_dim={state_dim}, action_dim={action_dim}")
        
        # Initialize DDPG agent with improved parameters
        self.agent = FLDDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[512, 256],
            buffer_size=10000,  # 𝒟 = 10000
            batch_size=16,  # 𝑇ₘₐₓ = 16
            gamma=0.95,  # γ_b = 0.95
            tau=0.00001,  # τ_tc and τ_ta = 0.00001
            actor_lr=0.001,  # μ_ta = 0.001
            critic_lr=0.001,  # μ_tc = 0.001
            lr_decay_rate=0.00001,  # λ_tc and λ_ta = 0.00001
            min_lr=1e-6,  # Not explicitly in the table, but might be useful
        )
        
        # self.agent = DDPGagent(
        #     num_states= state_dim,
        #     num_actions= action_dim,
        #     hidden_size_1= 512,
        #     hidden_size_2= 256,
        #     max_memory_size= 10000,
        #     disc_fact= 0.95,
        #     tau= 0.00001,
        #     actor_learning_rate= 0.001,
        #     critic_learning_rate= 0.001,
        #     lr_decay= 0.00001,
        #     min_lr= 1e-6
        # )

        
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
        # Create models directory if it doesn't exist
        models_folder = 'models'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save to the models folder with appropriate filename
        filepath = os.path.join(models_folder, f'{checkpoint_type}_model.pth')
        torch.save(checkpoint, filepath)

    
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
        plt.plot(self.metrics['avg_peb_values'], label='Avg PEB')
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
        
        plt_folder = 'plots'
        plt.tight_layout()
        plt.savefig(os.path.join(plt_folder,'training_progress.png'))
        plt.close()
    
    def train(self, num_episodes, max_steps, target_peb):
        print("Starting training...")
        # Nb = self.eng.get_Nb(self.sim)
        
        epsilon_start = 1.0
        epsilon_en = 0.01
        epsilon_decay_rate = 0.995
        
        peb_ae = [[] for _ in range(300)]
        prev_peb = 12
        avg_rate = []
        rate_values = [[] for _ in range(300)]
        power_values = [[] for _ in range(300)]
        avg_power = []
        # first_phi_val = []
        for episode in range(num_episodes):
            episode_start_time = time.time()
            # Reset environment
            matlab_state = self.eng.reset(self.sim)
            # first_phi_val.append(self.eng.initializephi())
            state = self.process_state(matlab_state)
            episode_reward = 0
            episode_losses = {'actor': [], 'critic': []}
            
            # Initialize PEB tracking for this episode
            initial_peb = 100
            min_peb_in_episode = initial_peb
            max_peb_in_episode = -1
            peb_values_in_episode = []
            
            # Initialize episode precoder
            step_counter = 0

            if(episode!=0):
                self.agent.decay_learning_rates()

            
            for step in range(max_steps):

                value = np.random.uniform(0,1)
                if (value < epsilon_start):
                    explore = True
                else:
                    explore = False
                # explore = False

                step_counter = step_counter + 1
                # Select action with exploration
                action = self.agent.select_action(state, explore, epsilon_start) 
                # action = (action+1)/2
                # Convert and execute action
                matlab_action = matlab.double(action.tolist())
                
                # current_peb = float(self.eng.calculatePerformanceMetrics(self.sim))
                next_matlab_state, reward, current_peb, rate, power, done = self.eng.step(self.sim, matlab_action, nargout=6)
                if(step==1):
                    initial_peb = current_peb
                            
                rate_values[episode].append(float(np.real(rate)))
                power_values[episode].append(float(power))
                peb_ae[episode].append(current_peb)
                # Track PEB values
                peb_values_in_episode.append(current_peb)
                min_peb_in_episode = min(min_peb_in_episode, current_peb)
                max_peb_in_episode = max(max_peb_in_episode, current_peb)
                
                # Process step results
                next_state = self.process_state(next_matlab_state)
                done = bool(done)
                
                # Store transition and update networks
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                actor_loss, critic_loss = self.agent.update()
                
                # Track step metrics
                episode_reward += reward
                if actor_loss is not None:
                    episode_losses['actor'].append(actor_loss)
                    episode_losses['critic'].append(critic_loss)
                
                state = next_state
                
                if done:
                    # self.eng.reset(self.sim)
                    break

            epsilon_start = max(epsilon_en, epsilon_start*epsilon_decay_rate)

            episode_time = time.time() - episode_start_time
            print(f"Episode {episode+1} completed in {episode_time:.2f} seconds")
            # np.savetext('phi_Vals', first_phi_val, fmt='%f')
            
            # Calculate average PEB for the episode
            avg_peb_in_episode = sum(peb_values_in_episode) / len(peb_values_in_episode)
            last_peb_in_episode = current_peb
            avg_rate = np.mean(rate_values[episode])
            avg_power = np.mean(power_values[episode])

            # Normalize reward
            episode_reward = episode_reward/step_counter


            # Store PEB metrics
            if 'initial_peb_values' not in self.metrics:
                self.metrics.update({
                    'initial_peb_values': [],
                    'min_peb_values': [],
                    'max_peb_values': [],
                    'avg_peb_values': [],
                    'last_peb_values': [],
                    'avg_rate': [],
                    'avg_power': [],
                    'exp_unexp_ratio': []
                })
            
            self.metrics['initial_peb_values'].append(initial_peb)
            self.metrics['min_peb_values'].append(min_peb_in_episode)
            self.metrics['max_peb_values'].append(max_peb_in_episode)
            self.metrics['avg_peb_values'].append(avg_peb_in_episode)
            self.metrics['last_peb_values'].append(last_peb_in_episode)
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['avg_rate'].append(avg_rate)
            self.metrics['avg_power'].append(avg_power)
            self.metrics['peb_values'].append(current_peb)  # For backward compatibility
            if episode_losses['actor']:
                self.metrics['actor_losses'].append(np.mean(episode_losses['actor']))
                self.metrics['critic_losses'].append(np.mean(episode_losses['critic']))
            self.metrics['learning_rates'].append(self.agent.current_actor_lr)
            
            # Update best metrics and save checkpoints
            if episode_reward > self.best_metrics['reward']:
                self.best_metrics['reward'] = episode_reward
                self.best_metrics['reward_episode'] = episode
                self.save_checkpoint(episode, self.metrics, 'best_reward')
            
            # When updating best metrics
            if min_peb_in_episode < self.best_metrics['peb']:
                self.best_metrics['peb'] = min_peb_in_episode
                self.best_metrics['peb_episode'] = episode
                self.save_checkpoint(episode, self.metrics, 'best_peb')
            
            # Print progress
            pzc = self.eng.getpebzero(self.sim)

            if episode % 1 == 0:
                self.plot_training_progress()
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Reward: {episode_reward:.3f}")
                print(f"Initial PEB: {initial_peb:.6f}")
                # print(f"Min PEB: {min_peb_in_episode:.6f}")
                # print(f"Max PEB: {max_peb_in_episode:.6f}")
                print(f"Avg PEB: {avg_peb_in_episode:.6f}")
                print(f"Last PEB: {last_peb_in_episode:.6f}")
                print(f"Best PEB (all episodes): {self.best_metrics['peb']:.6f}")
                print(f"peb was zero: {pzc} times")
                print(f"Learning Rate: {self.agent.get_current_actor_lr():.12f}")
                print(f"Buffer Size: {len(self.agent.replay_buffer)}")
                print("-" * 50)
            
            # Early stopping check
            if episode > 1000 and np.mean(self.metrics['min_peb_values'][-1000:]) < target_peb:
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
                action = self.agent.select_action(state, explore=False, epsilon=0.0)
                matlab_action = matlab.double(action.tolist())
                next_state, reward, done = self.eng.step(self.sim, matlab_action, nargout=3)
                
                state = np.array(next_state)
                episode_reward += float(abs(reward))
                step += 1
            
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.3f}")
    
    def close(self):
        """Clean up MATLAB engine"""
        # self.eng.eval("delete(gcp('nocreate'));", nargout=0)
        self.eng.quit()

if __name__ == "__main__":
    # Create trainer instance
    trainer = RISISACTrainer()
    print(type(trainer.sim))
    
    
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

    # plt.figure(figsize=(15, 10))

    # # Plot PEB metrics
    # plt.subplot(2, 2, 1)
    # plt.plot(trainer.metrics['initial_peb_values'], label='Initial PEB')
    # plt.plot(trainer.metrics['last_peb_values'], label='Last PEB')
    # plt.title('Initial vs Final PEB per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('PEB')
    # plt.legend()

    # plt.subplot(2, 2, 2)
    # plt.plot(trainer.metrics['min_peb_values'], label='Min PEB')
    # plt.plot(trainer.metrics['avg_peb_values'], label='Avg PEB')
    # plt.plot(trainer.metrics['max_peb_values'], label='Max PEB')
    # plt.title('PEB Statistics per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('PEB')
    # plt.legend()
    
    plt_folder = 'plots'
    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)
    
    # Plot training rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(plt_folder,'training_rewards.png'))
    # plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, metrics['last_peb_values'])
    plt.title('Performance Error Bound (PEB) over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('PEB Value')
    plt.grid(True)
    plt.savefig(os.path.join(plt_folder,'peb_values.png'))
    # plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(episodes, metrics['avg_rate'])
    plt.title('Average rate per epsiode')
    plt.xlabel('Episode')
    plt.ylabel('Rate(Bits/s/hz)')
    plt.grid(True)
    plt.savefig(os.path.join(plt_folder,'Rate_per_episode'))
    # plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(episodes, metrics['avg_power'])
    plt.title('Average power per episode')
    plt.xlabel('Episode')
    plt.ylabel('Power(db)')
    plt.grid(True)
    plt.savefig(os.path.join(plt_folder,'power_per_episode'))
    trainer.close()