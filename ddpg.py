import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, state):
        return self.network(state)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (torch.FloatTensor(state), torch.FloatTensor(action),
                torch.FloatTensor(reward), torch.FloatTensor(next_state))
    
    def __len__(self):
        return len(self.buffer)

class FLDDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=256, buffer_size=1000000,
                 batch_size=64, gamma=0.99, tau=0.001, actor_lr=1e-4, critic_lr=1e-3):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)       
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration parameters
        self.noise_scale = 1.0
        self.noise_decay = 0.995
        self.min_noise = 0.1
        
    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().squeeze()
        
        if explore:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
            
        return action
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size * 3:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            target_q = reward_batch + self.gamma * self.critic_target(next_state_batch, next_actions)
        
        current_q = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def train(self, env, max_episodes, max_steps):
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select action
                action = self.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state)
                
                # Update networks
                self.update()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            print(f"Episode {episode + 1}, Reward: {episode_reward}")