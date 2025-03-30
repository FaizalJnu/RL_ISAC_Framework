import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import copy

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        super(ActorNetwork, self).__init__()
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
    
    def get_action(self, state):
        action = self.forward(state)
        return action

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
    
# class ReplayBuffer:
#     def __init__(self, capacity, state_dim):
#         self.buffer = deque(maxlen=capacity)
#         self.state_dim = state_dim
        
#     def push(self, state, action, reward, next_state, done):
#         state = np.squeeze(np.array(state))
#         next_state = np.squeeze(np.array(next_state))
#         assert state.shape == (self.state_dim,), f"State shape mismatch: {state.shape}"
#         assert next_state.shape == (self.state_dim,), f"Next state shape mismatch: {next_state.shape}"
        
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def __len__(self):
#         return len(self.buffer)
        
#     def sample(self, batch_size):
#         if len(self.buffer) < batch_size:
#             raise ValueError(f"Not enough samples in buffer ({len(self.buffer)}) to sample batch of {batch_size}")
            
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*batch)

#         state = np.array(state)
#         next_state = np.array(next_state)
#         action = np.array(action)
#         reward = np.array(reward).reshape(-1, 1)
#         done = np.array(done).reshape(-1, 1)
        
#         assert state.shape == (batch_size, self.state_dim), f"State shape mismatch: {state.shape}"
#         assert next_state.shape == (batch_size, self.state_dim), f"Next state shape mismatch: {next_state.shape}"
        
#         return (torch.FloatTensor(state), 
#                 torch.FloatTensor(action),
#                 torch.FloatTensor(reward), 
#                 torch.FloatTensor(next_state),
#                 torch.FloatTensor(done))

class SumTree:
    """
    A binary sum-tree data structure for efficient sampling based on priorities.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure
        
        # Initialize with None instead of zeros
        self.data = np.array([None] * capacity, dtype=object)  # Experience data
        
        self.size = 0  # Current size
        self.next_idx = 0  # Next index to write

    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample based on priority value s"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return sum of all priorities"""
        return self.tree[0]

    def add(self, priority, data):
        """Add new experience with priority"""
        idx = self.next_idx + self.capacity - 1
        self.data[self.next_idx] = data
        self.update(idx, priority)

        self.next_idx = (self.next_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, priority):
        """Update priority of existing experience"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get experience based on priority value s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        # Make sure data_idx is valid
        if data_idx < 0 or data_idx >= self.size:
            # Use a valid index instead
            data_idx = max(0, min(self.size - 1, data_idx))
        
        # Check if the data is a valid experience tuple
        if not isinstance(self.data[data_idx], tuple):
            print(f"Warning: Experience at index {data_idx} is not a tuple: {self.data[data_idx]}")
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            state_dim: Dimension of state space
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time we sample
            epsilon: Small constant to ensure non-zero priorities
        """
        self.tree = SumTree(capacity)
        self.state_dim = state_dim
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        state = np.squeeze(np.array(state))
        next_state = np.squeeze(np.array(next_state))
        assert state.shape == (self.state_dim,), f"State shape mismatch: {state.shape}"
        assert next_state.shape == (self.state_dim,), f"Next state shape mismatch: {next_state.shape}"
        
        experience = (state, action, reward, next_state, done)
        # New experiences get max priority to ensure they're sampled at least once
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """Sample batch_size experiences based on priorities"""
        batch = []
        indices = []
        priorities = []
        
        # Check if buffer has enough samples
        if len(self) < batch_size:
            raise ValueError(f"Not enough experiences in buffer. Current size: {len(self)}")
        
        segment = self.tree.total() / batch_size
        
        # Increase beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            
            # Verify experience is a valid tuple
            if not isinstance(experience, tuple) or len(experience) != 5:
                print(f"Invalid experience at index {idx}: {experience}")
                # Create a default experience
                zero_state = np.zeros(self.state_dim)
                experience = (zero_state, 0, 0.0, zero_state, False)
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(experience)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        sampling_probabilities = np.clip(sampling_probabilities, 1e-8, 1.0)
        weights = (len(self.tree.data) * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to numpy arrays
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward).reshape(-1, 1)
        done = np.array(done).reshape(-1, 1)
        weights = np.array(weights).reshape(-1, 1)
        
        # Verify shapes
        assert state.shape == (batch_size, self.state_dim), f"State shape mismatch: {state.shape}"
        assert next_state.shape == (batch_size, self.state_dim), f"Next state shape mismatch: {next_state.shape}"
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
            torch.FloatTensor(weights),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            # Add epsilon to ensure non-zero priority
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        """Return current size of buffer"""
        return self.tree.size

class FLDDPG:
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], buffer_size=10000,
                 batch_size=16, gamma=0.95, tau=0.00001, actor_lr=1e-3, critic_lr=1e-3,
                 lr_decay_rate=0.00001, min_lr=1e-6):
        
        #? Selecting GPU based on device
        if(torch.backends.mps.is_available()):
            print("M1 GPU is available")
            self.device = torch.device("mps")
        else:
            print("Nvidia GPU is available")
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
        # self.replay_buffer = ReplayBuffer(buffer_size, state_dim)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            alpha=0.6,  # Priority exponent
            beta=0.4,   # Initial importance sampling weight
            beta_increment=0.001  # Beta annealing rate
        )

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
        
    def select_action(self, state, explore, epsilon):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                action = self.actor.get_action(state)
                noise_scale = epsilon * np.random.normal(0, 1, size=action.shape)
                action = torch.clamp(action + torch.FloatTensor(noise_scale).to(self.device), -1, 1)
            else:
                action = self.actor(state)
        return action.cpu().numpy().squeeze()

    # def update(self):
    #     if len(self.replay_buffer) < self.batch_size * 3:
    #         return 0, 0
    #     state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
    #     state_batch = state_batch.to(self.device)
    #     action_batch = action_batch.to(self.device)
    #     reward_batch = reward_batch.to(self.device)
    #     next_state_batch = next_state_batch.to(self.device)
    #     done_batch = done_batch.to(self.device)
        
    #     with torch.no_grad():
    #         next_actions = self.target_actor(next_state_batch)
    #         target_q = reward_batch + self.gamma * self.target_critic(next_state_batch, next_actions)
        
    #     current_q = self.critic(state_batch, action_batch)
    #     critic_loss = torch.nn.MSELoss()(current_q, target_q)

    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
    #     self.critic_optimizer.step()

    #     actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
    #     self.actor_optimizer.step()

    #     self.target_actor.soft_update()
    #     self.target_critic.soft_update()
        
    #     return actor_loss.item(), critic_loss.item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size * 3:
            return 0, 0
        
        # Sample batch with importance sampling weights and indices
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = self.replay_buffer.sample(self.batch_size)
        
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        weights = weights.to(self.device)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch)
            target_q = reward_batch + self.gamma * self.target_critic(next_state_batch, next_actions)
        
        current_q = self.critic(state_batch, action_batch)
        
        # Calculate TD errors for updating priorities
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        
        # Apply importance sampling weights to the critic loss
        critic_loss = (weights * (current_q - target_q)**2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Actor loss remains the same (no need for importance sampling here)
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        self.target_actor.soft_update()
        self.target_critic.soft_update()
        
        return actor_loss.item(), critic_loss.item()
