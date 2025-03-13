import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import copy

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], epsilon=0.1):
        super(ActorNetwork, self).__init__()
        
        self.epsilon = epsilon
        layers = []
        prev_dim = state_dim
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
        return self.policy_network(state)
    
    def get_action_with_noise(self, state):
        action = self.forward(state)
        noise = torch.randn_like(action) * self.epsilon
        noisy_action = action + noise
        return torch.clamp(noisy_action, -1.0, 1.0)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        super(CriticNetwork, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0])
        )
        layers = []
        prev_dim = hidden_dims[0] + action_dim
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_network = nn.Sequential(*layers)
        
    def forward(self, state, action):
        state_features = self.state_layer(state)
        combined = torch.cat([state_features, action], dim=1)
        q_value = self.q_network(combined)
        
        return q_value

class TargetActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], tau=0.001):
        super(TargetActorNetwork, self).__init__()
        
        self.tau = tau
        self.online_network = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = copy.deepcopy(self.online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def forward(self, state):
        return self.target_network(state)
    
    def soft_update(self):
        for target_param, online_param in zip(self.target_network.parameters(), 
                                            self.online_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_online_network(self):
        return self.online_network
    
    def get_target_network(self):
        return self.target_network

class TargetCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], tau=0.001):
        super(TargetCriticNetwork, self).__init__()
        
        self.tau = tau
        self.online_network = CriticNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = copy.deepcopy(self.online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def forward(self, state, action):
        return self.target_network(state, action)
    
    def soft_update(self):
        for target_param, online_param in zip(self.target_network.parameters(), 
                                            self.online_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_online_network(self):
        return self.online_network
    
    def get_target_network(self):
        return self.target_network
    
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim 
        
    def push(self, state, action, reward, next_state):
        state = np.squeeze(np.array(state))
        next_state = np.squeeze(np.array(next_state))
        assert state.shape == (self.state_dim,), f"State shape mismatch: {state.shape}"
        assert next_state.shape == (self.state_dim,), f"Next state shape mismatch: {next_state.shape}"
        
        self.buffer.append((state, action, reward, next_state))
    
    def __len__(self):
        return len(self.buffer)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)

        state = np.array([np.squeeze(s) for s in state])
        next_state = np.array([np.squeeze(ns) for ns in next_state])
        
        action = np.array(action)
        reward = np.array(reward).reshape(-1, 1)
        
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
        if(torch.backends.mps.is_available()):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda")
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
        self.actor = self.target_actor.get_online_network()
        self.critic = self.target_critic.get_online_network()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim)

        self.epsilon = 0.5 
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.initial_actor_lr = actor_lr
        self.initial_critic_lr = critic_lr
        self.lr_decay_rate = lr_decay_rate
        self.min_lr = min_lr
        self.current_actor_lr = actor_lr
        self.current_critic_lr = critic_lr

        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.actor_optimizer, 
            gamma=lr_decay_rate
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.critic_optimizer, 
            gamma=lr_decay_rate
        )

    def decay_learning_rates(self):
        self.actor_scheduler.step()
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr)
        self.current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
        self.critic_scheduler.step()
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr)
        self.current_critic_lr = self.critic_optimizer.param_groups[0]['lr']
        
    def select_action(self, state, explore):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                # Even the noise here is set to decay over time to lead to stability
                action = self.actor.get_action_with_noise(state)
                noise_scale = self.epsilon * np.random.normal(0, 1, size=action.shape)
                self.epsilon = max(self.epsilon_decay*self.epsilon, self.min_epsilon)
                action = torch.clamp(action + torch.FloatTensor(noise_scale).to(self.device), -1, 1)
            else:
                action = self.actor(state)
        return action.cpu().numpy().squeeze()

    def update(self):
        if len(self.replay_buffer) < self.batch_size * 3:
            return 0, 0
        state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample(self.batch_size)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch)
            target_q = reward_batch + self.gamma * self.target_critic(next_state_batch, next_actions)
        
        current_q = self.critic(state_batch, action_batch)
        critic_loss = torch.nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.target_actor.soft_update()
        self.target_critic.soft_update()
        
        return actor_loss.item(), critic_loss.item()