import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import random
from collections import deque

import torch.autograd
import torch.optim as optim
import torch.nn as nn

"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, 
                 min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.low          = 0
        self.high         = 2*np.pi
        self.action_dim = action_dim
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
     
class Replay:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        print("we are finally here")

        for experience in batch:
            state, action, reward, next_state = experience
            #print(state.shape)
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            #done_batch.append(done)
        #print(state_batch)
        #print(torch.FloatTensor(state_batch))
        
        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, learning_rate):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear2(F.relu(self.linear1(state))))
        x = torch.tanh(self.linear3(x))
        if torch.isnan(x).any():
            print("NaN detected in actor output!")
            print(f"Input state min/max: {state.min().item()}, {state.max().item()}")
            # You could also print weights to see if they've exploded
            print(f"Layer 1 weights min/max: {self.linear1.weight.min().item()}, {self.linear1.weight.max().item()}")
        return x



class Critic(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)


    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #print(state.shape)
        #print(action.shape)
        x = torch.cat([state, action], dim = 1)
        x = F.leaky_relu_(self.linear2(F.relu(self.linear1(x))))
        x = torch.tanh(self.linear3(x))

        return x

class DDPGagent:
    def __init__(self, num_states, num_actions, hidden_size_1, 
                 hidden_size_2, 
                 max_memory_size,
                 disc_fact, tau,
                 actor_learning_rate, 
                 critic_learning_rate,
                 lr_decay, min_lr):
        # Params
        if(torch.backends.mps.is_available()):
            print("M1 GPU is available")
            self.device = torch.device("mps")
        else:
            print("Nvidia GPU is available")
            self.device = torch.device("cuda")
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.disc_fact = disc_fact
        self.tau = tau
        self.min_lr = min_lr

        self.noise = OUNoise(action_dim=num_actions, 
                        mu=0.0, 
                        theta=0.15, 
                        max_sigma=0.3, 
                        min_sigma=0.1, 
                        decay_period=10000)


        # Networks
        self.actor_eval = Actor(num_states, hidden_size_1, hidden_size_2, num_actions, actor_learning_rate).to(self.device)
        self.actor_target = Actor(num_states, hidden_size_1, hidden_size_2, num_actions, actor_learning_rate).to(self.device)
        self.critic_eval = Critic(num_states + num_actions, hidden_size_1, hidden_size_2, 1).to(self.device)
        self.critic_target = Critic(num_actions+num_states, hidden_size_1, hidden_size_2, 1).to(self.device)

        self.replay_buffer = Replay(
            max_size=max_memory_size
        )
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Replay(max_memory_size)       
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor_eval.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=critic_learning_rate)

        decay_gamma = 1.0 - lr_decay  # Convert your decay rate to scheduler's gamma
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.actor_optimizer,
            gamma=decay_gamma,
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.critic_optimizer,
            gamma=decay_gamma
        )

        self.current_actor_lr = actor_learning_rate
        self.current_critic_lr = critic_learning_rate

                # In your optimizer update step
        torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_eval.parameters(), max_norm=1.0)

    def get_current_actor_lr(self):
        return self.actor_optimizer.param_groups[0]['lr']

    def select_action(self, state, explore, t=0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_eval.forward(state)
            action = action.cpu().numpy().squeeze()
            
            if explore:
                noisy_action = self.noise.get_action(action, t)
                return noisy_action
            else:
                return action
    
    def decay_learning_rates(self):
        # Step the schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # Enforce minimum learning rate
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr)
        self.current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
        # print("Current actor learning rate:", self.current_actor_lr)  # Debug print
        
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr)
        self.current_critic_lr = self.critic_optimizer.param_groups[0]['lr']  

    def update(self, batch_size):
        print(len(self.memory)) 
        if len(self.memory) < batch_size:
            return None, None  # Skip update if not enough data
        
        states, actions, rewards, next_states = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        Qvals = self.critic_eval.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards.clone()
        for i in range(len(rewards)):
          Qprime[i] = rewards[i] + self.disc_fact* next_Q[i]
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic_eval.forward(states, self.actor_eval.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
        return critic_loss.item(), policy_loss.item()