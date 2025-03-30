import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim=161, action_dim=64, n_latent=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
            nn.ReLU()
        )
        
        # Actor network - outputs discrete action probabilities for each RIS element
        # For 2-bit quantization (4 possible values per element)
        self.actor = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, action_dim * 4),  # 4 possible phase shifts per element
        )
        
        # Critic network - estimates value function
        self.critic = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, 1)
        )
        
    def forward(self, state):
        features = self.feature_extractor(state)
        
        # Reshape actor output to (batch_size, action_dim, 4) for discrete actions
        action_probs = self.actor(features).view(-1, 64, 4)
        action_probs = torch.softmax(action_probs, dim=2)
        
        # Value function
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action(self, state, action=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self(state)
        
        # Create a distribution for each RIS element
        dist = [Categorical(probs=action_probs[0, i]) for i in range(64)]
        
        if action is None:
            # Sample action for each RIS element
            action = torch.stack([dist[i].sample() for i in range(64)])
        
        # Calculate log probabilities
        log_probs = torch.stack([dist[i].log_prob(action[i]) for i in range(64)])
        
        return action.detach().numpy(), value.detach().numpy(), log_probs.sum()

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(memory.states))
        old_actions = torch.LongTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs))
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = [Categorical(action_probs[:, i]) for i in range(64)]
            
            # Calculate log probabilities
            logprobs = torch.stack([dist[i].log_prob(old_actions[:, i]) for i in range(64)])
            logprobs = logprobs.sum(dim=0)  # Sum across all RIS elements
            
            # Finding the ratio (π_θ / π_θ_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
