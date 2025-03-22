import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.memsize = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.memsize, *input_shape))
        self.new_state_memory = np.zeros((self.memsize, *input_shape))
        self.action_memory = np.zeros((self.memsize, n_actions))
        self.reward_memory = np.zeros(self.memsize)
        self.terminal_memory = np.zeros(self.memsize, dtype=np.bool)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.memsize
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done #value of initial state is 0, determining whether we are starting
        
        
        # self.new_state_memory[index] = new_state
        
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.memsize)
        
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
    
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims = 512, fc2_dims = 512,
                 name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # self.n_Actions = n_Actions
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_ddpg.h5')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
        
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        
        q = self.q(action_value)
        
        return q

class ActorNetwork(keras.model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor', 
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_ddpg.h5')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        
        # since we don't want a null activation function, we use tanh
        self.mu = Dense(self.n_actions, activation='tanh')
        
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        
        mu = self.mu(prob)
        
        
        