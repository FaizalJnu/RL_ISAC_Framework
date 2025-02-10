import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import copy

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], epsilon=0.1):
        """
        Initialize the Actor Network for DDPG
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (list): Dimensions of hidden layers
            epsilon (float): Exploration rate for ε-greedy strategy
        """
        super(ActorNetwork, self).__init__()
        
        self.epsilon = epsilon
        layers = []
        prev_dim = state_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim) 
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.policy_network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass implementing equation (21): a_tm = μ(s_tm|ω_π)
        
        Args:
            state (torch.Tensor): Current state tensor
        Returns:
            torch.Tensor: Action without exploration noise
        """
        return self.policy_network(state)
    
    def get_action_with_noise(self, state):
        """
        Implementation of equation (22): a_tm = μ(s_tm|ω_π) + ε*n_e
        
        Args:
            state (torch.Tensor): Current state tensor
        Returns:
            torch.Tensor: Action with exploration noise
        """
        # Get the base action from the policy network
        action = self.forward(state)
        
        # Add exploration noise
        noise = torch.randn_like(action) * self.epsilon
        noisy_action = action + noise
        
        # Clip the actions to be in [-1, 1] range
        return torch.clamp(noisy_action, -1.0, 1.0)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        """
        Initialize the Critic Network for DDPG
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (list): Dimensions of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        # First layer processes just the state
        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0])
        )
        
        # Process state + action together
        layers = []
        # Input dimension is first hidden dim + action dim (concatenated)
        prev_dim = hidden_dims[0] + action_dim
        
        # Add remaining hidden layers
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final output layer for Q-value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_network = nn.Sequential(*layers)
        
    def forward(self, state, action):
        """
        Forward pass to compute Q(s, π(s|a')|θ)
        
        Args:
            state (torch.Tensor): Current state tensor
            action (torch.Tensor): Action tensor from actor network
        Returns:
            torch.Tensor: Q-value for the state-action pair
        """
        # Process state first
        state_features = self.state_layer(state)
        
        # Concatenate state features and action
        combined = torch.cat([state_features, action], dim=1)
        
        # Compute Q-value
        q_value = self.q_network(combined)
        
        return q_value

class TargetActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], tau=0.001):
        """
        Initialize the Target Actor Network for DDPG
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (list): Dimensions of hidden layers
            tau (float): Soft update factor (0 < tau << 1)
        """
        super(TargetActorNetwork, self).__init__()
        
        self.tau = tau
        
        # Create the online actor network
        self.online_network = ActorNetwork(state_dim, action_dim, hidden_dims)
        
        # Create the target network as a copy of the online network
        self.target_network = copy.deepcopy(self.online_network)
        
        # Freeze target network parameters (they'll only be updated through soft updates)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def forward(self, state):
        """
        Forward pass using the target network
        
        Args:
            state (torch.Tensor): Current state tensor
        Returns:
            torch.Tensor: Target action
        """
        return self.target_network(state)
    
    def soft_update(self):
        """
        Implements equation (23): θ'_π = τ*θ_π + (1-τ)*θ'_π
        Performs soft update of target network parameters using online network parameters
        """
        for target_param, online_param in zip(self.target_network.parameters(), 
                                            self.online_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_online_network(self):
        """
        Returns the online actor network for training
        """
        return self.online_network
    
    def get_target_network(self):
        """
        Returns the target network for generating target actions
        """
        return self.target_network

class TargetCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], tau=0.001):
        """
        Initialize the Target Critic Network for DDPG
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (list): Dimensions of hidden layers
            tau (float): Soft update factor (0 < tau << 1)
        """
        super(TargetCriticNetwork, self).__init__()
        
        self.tau = tau
        
        # Create the online critic network
        self.online_network = CriticNetwork(state_dim, action_dim, hidden_dims)
        
        # Create the target network as a copy of the online network
        self.target_network = copy.deepcopy(self.online_network)
        
        # Freeze target network parameters (they'll only be updated through soft updates)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def forward(self, state, action):
        """
        Forward pass using the target network to compute Q(s_{t+1}, π(s_{t+1}|a')|θ_Q')
        
        Args:
            state (torch.Tensor): Next state tensor
            action (torch.Tensor): Action tensor from target actor network
        Returns:
            torch.Tensor: Target Q-value
        """
        return self.target_network(state, action)
    
    def soft_update(self):
        """
        Implements soft update of target network parameters using online network parameters
        θ_Q' = τ_Q*θ_Q + (1-τ_Q)*θ_Q'
        """
        for target_param, online_param in zip(self.target_network.parameters(), 
                                            self.online_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_online_network(self):
        """
        Returns the online critic network for training
        """
        return self.online_network
    
    def get_target_network(self):
        """
        Returns the target network for generating target Q-values
        """
        return self.target_network
    
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim  # Define state_dim to ensure consistency
        
    def push(self, state, action, reward, next_state):
        # Normalize state shapes to 1D arrays
        state = np.squeeze(np.array(state))
        next_state = np.squeeze(np.array(next_state))
        
        # Optional debugging to verify shapes
        # print(f"Pushing state shape: {state.shape}, next_state shape: {next_state.shape}")
        
        # Ensure states are the correct dimension
        assert state.shape == (self.state_dim,), f"State shape mismatch: {state.shape}"
        assert next_state.shape == (self.state_dim,), f"Next state shape mismatch: {next_state.shape}"
        
        self.buffer.append((state, action, reward, next_state))
    
    def __len__(self):
        return len(self.buffer)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        
        # Normalize state shapes - ensure all states are 1D arrays of shape (138,)
        state = np.array([np.squeeze(s) for s in state])  # Will convert (1,138) to (138,)
        next_state = np.array([np.squeeze(ns) for ns in next_state])
        
        action = np.array(action)
        reward = np.array(reward).reshape(-1, 1)
        
        # Assertions for shape checking
        assert state.shape == (batch_size, self.state_dim), f"State shape mismatch: {state.shape}"
        assert next_state.shape == (batch_size, self.state_dim), f"Next state shape mismatch: {next_state.shape}"
        assert action.shape[0] == batch_size, f"Action batch size mismatch: {action.shape[0]} != {batch_size}"
        assert reward.shape[0] == batch_size, f"Reward batch size mismatch: {reward.shape[0]} != {batch_size}"
        
        return (torch.FloatTensor(state), 
                torch.FloatTensor(action),
                torch.FloatTensor(reward), 
                torch.FloatTensor(next_state))

class FLDDPG:
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], buffer_size=10000,
                 batch_size=16, gamma=0.95, tau=0.00001, actor_lr=1e-3, critic_lr=1e-3,
                 lr_decay_rate=0.995, min_lr=1e-6):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize networks using our implementations
        self.target_actor = TargetActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            tau=tau
        ).to(self.device)
        
        self.target_critic = TargetCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            tau=tau
        ).to(self.device)
        
        # Get references to online networks
        self.actor = self.target_actor.get_online_network()
        self.critic = self.target_critic.get_online_network()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim)
        
        # Exploration parameters
        self.epsilon = 0.1  # Using epsilon from our actor network implementation
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.initial_actor_lr = actor_lr
        self.initial_critic_lr = critic_lr
        self.lr_decay_rate = lr_decay_rate
        self.min_lr = min_lr
        self.current_actor_lr = actor_lr
        self.current_critic_lr = critic_lr

        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, 
            gamma=lr_decay_rate
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, 
            gamma=lr_decay_rate
        )

    def decay_learning_rates(self):
        """Decay learning rates with minimum threshold"""
        if self.current_actor_lr > self.min_lr:
            self.actor_scheduler.step()
            self.current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            
        if self.current_critic_lr > self.min_lr:
            self.critic_scheduler.step()
            self.current_critic_lr = self.critic_optimizer.param_groups[0]['lr']
        
    def select_action(self, state, explore=True):
        """Select action using the actor network with optional exploration"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                # Use the actor's exploration method with decaying noise
                action = self.actor.get_action_with_noise(state)
                # Add additional Gaussian noise that decays over time
                noise_scale = self.epsilon * np.random.normal(0, 1, size=action.shape)
                action = torch.clamp(action + torch.FloatTensor(noise_scale).to(self.device), -1, 1)
            else:
                # Use deterministic action
                action = self.actor(state)
        return action.cpu().numpy().squeeze()

    def update(self):
        """Update the networks using experience replay"""
        if len(self.replay_buffer) < self.batch_size * 3:
            return 0, 0  # Return losses for monitoring
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch)
            target_q = reward_batch + self.gamma * self.target_critic(next_state_batch, next_actions)
        
        current_q = self.critic(state_batch, action_batch)
        critic_loss = torch.nn.MSELoss()(current_q, target_q)
        
        # Gradient clipping for critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update actor with delayed policy updates
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        # Gradient clipping for actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.target_actor.soft_update()
        self.target_critic.soft_update()
        
        return actor_loss.item(), critic_loss.item()

    def train(self, env, max_episodes, max_steps):
        """Train the agent in the environment"""
        best_reward = float('-inf')
        best_peb = float('inf')
        rewards_history = []
        peb_history = []
        actor_losses = []
        critic_losses = []
        
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            episode_peb = 0
            episode_actor_losses = []
            episode_critic_losses = []
            
            for step in range(max_steps):
                # Select action with exploration
                action = self.select_action(state, explore=True)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state)
                
                # Update networks and get losses
                actor_loss, critic_loss = self.update()
                if actor_loss != 0:
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
                
                episode_reward += reward
                episode_peb = info.get('peb', 0) 
                state = next_state
                
                if done:
                    break
            
            # Decay learning rates
            self.decay_learning_rates()
            
            # Store history
            rewards_history.append(episode_reward)
            peb_history.append(episode_peb)
            if episode_actor_losses:  # Only if updates occurred
                actor_losses.append(np.mean(episode_actor_losses))
                critic_losses.append(np.mean(episode_critic_losses))
            
            # Update best metrics
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'reward': best_reward,
                    'episode': episode
                }, 'best_reward_model.pth')
            
            if episode_peb < best_peb:
                best_peb = episode_peb
                torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'peb': best_peb,
                    'episode': episode
                }, 'best_peb_model.pth')
            
            # Print progress every N episodes
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_peb = np.mean(peb_history[-100:])
                avg_actor_loss = np.mean(actor_losses[-100:]) if actor_losses else 0
                avg_critic_loss = np.mean(critic_losses[-100:]) if critic_losses else 0
                
                print(f"Episode {episode + 1}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average PEB: {avg_peb:.4f}")
                print(f"Best PEB: {best_peb:.4f}")
                print(f"Actor Loss: {avg_actor_loss:.4f}")
                print(f"Critic Loss: {avg_critic_loss:.4f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print(f"Learning Rate: {self.current_actor_lr:.6f}")
                print("-" * 50)
        
        return rewards_history, peb_history, actor_losses, critic_losses