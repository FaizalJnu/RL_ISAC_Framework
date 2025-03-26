import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.memsize = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.memsize, *input_shape))
        self.new_state_memory = np.zeros((self.memsize, *input_shape))
        self.action_memory = np.zeros((self.memsize, n_actions))
        self.reward_memory = np.zeros(self.memsize)
        self.terminal_memory = np.zeros(self.memsize, dtype=np.bool_)
        
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
# import tensorflow.keras as keras
from tensorflow import keras
from keras import layers

if(tf.config.list_physical_devices('GPU')):
    tf.device('/device:GPU:0')
    print("GPU is available")
else:
    tf.device('/device:CPU:0')
    print("GPU is not available")


class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims = 512, fc2_dims = 512,
                 name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_ddpg.weights.h5')
        
        self.fc1 = layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = layers.Dense(self.fc2_dims, activation='relu')
        self.q = layers.Dense(1, activation=None)
        
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        
        q = self.q(action_value)
        
        return q

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, name='actor', 
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_ddpg.weights.h5')
        
        self.fc1 = layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = layers.Dense(self.fc2_dims, activation='relu')
        
        # since we don't want a null activation function, we use tanh
        self.mu = layers.Dense(self.n_actions, activation='tanh')
        
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        
        mu = self.mu(prob)
        
        return mu
    
from keras import optimizers

class Agent:
    def __init__(self, input_dims, n_actions, alpha=0.001, beta=0.002, env = None,
                 gamma=0.99, max_size=1000000, tau=0.005,
                 fc1_dims=400, fc2_dims=300, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = noise
        # self.max_action = env.action_space.high[0]
        # self.min_action = env.action_space.low[0]
        
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(n_actions = n_actions, name='critic')
        
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(n_actions=n_actions, name='target_critic')
        
        self.actor.compile(optimizer=optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=optimizers.Adam(learning_rate=beta))
        
        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def save_models(self):
        print("..saving models..")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        
    def load_models(self):
        print("...loading models...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    # def choose_action(self, observation, explore, evaluate=False):
    #     state = tf.convert_to_tensor([observation], dtype=tf.float32)
    #     actions = self.actor(state)
    #     if not evaluate or explore:
    #         actions += tf.random.normal(shape=[self.actor.n_actions], mean=0.0, stddev=self.noise)

    #     actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        
    #     return actions[0]

    def choose_action(self, observation, explore, epsilon):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action = self.actor(state)
        
        if explore:
            noise = tf.random.normal(shape=action.shape, mean=0.0, stddev=epsilon)
            action += noise
        
        return action[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                                self.actor.trainable_variables) 
        
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

import matlab.engine
import matplotlib.pyplot as plt
import os
# from utils import plot_learning_curve

class RISISACTrainer:
    def __init__(self):
        print("Starting training...")
        self.eng = matlab.engine.start_matlab()
        self.sim = self.eng.RISISAC_V2X_Sim()

        initial_state = self.eng.getState(self.sim)
        initial_state = np.array(initial_state).flatten()

        state_dim = initial_state.shape
        action_dim = 64

        self.agent = Agent(input_dims=state_dim, n_actions=action_dim)
        self.metrics = {}

    def process_state(self, matlab_state):
        return np.array(matlab_state).flatten()
    
    def train(self, num_episodes, max_steps):
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995

        rate_vals = [[] for _ in range(num_episodes)]

        power_vals = [[] for _ in range(num_episodes)]

        for episode in range(num_episodes):
            matlab_state = self.eng.getState(self.sim)
            state = self.process_state(matlab_state)
            reward = 0
            episode_reward = 0
            done = False
            episode_losses = {'actor': [], 'critic': []}

            initial_peb = float(self.eng.calculatePerformanceMetrics(self.sim))
            min_peb = initial_peb
            peb_values_in_episode = [initial_peb]
            rewards = [[] for _ in range(num_episodes)]
            step_counter = 0

            for step in range(max_steps):

                value = np.random.uniform(0,1)
                if value > epsilon:
                    explore = True
                else:
                    explore = False

                step_counter += 1
                action = self.agent.choose_action(state, explore, epsilon)

                matlab_action = matlab.double(action.numpy().tolist())

                next_matlab_state, reward, cpeb, rate, power, done = self.eng.step(self.sim, matlab_action, nargout=6)
                
                rate_vals[episode].append(rate)
                power_vals[episode].append(power)
                rewards[episode].append(reward)

                peb_values_in_episode.append(cpeb)
                min_peb = min(min_peb, cpeb)

                next_state = self.process_state(next_matlab_state)
                episode_reward += reward

                done = bool(done)

                self.agent.remember(state, action, reward, next_state, done)
                # actor_loss, critic_loss = self.agent.learn()

                if not explore:
                    self.agent.learn()

                state = next_state
                epsilon = max(epsilon*epsilon_decay, epsilon_min)

                if done:
                    break
            
            avg_rate = np.mean(rate_vals[episode])
            avg_power = np.mean(power_vals[episode])
            avg_reward = np.mean(rewards[episode])
            
                
            if 'initial_peb' not in self.metrics:
                self.metrics.update({
                    'initial_peb': [initial_peb],
                    'min_peb' : [min_peb],
                    'rate': [avg_rate],
                    'power': [avg_power],
                    'reward': [avg_reward]
                })
            
            self.metrics['initial_peb'].append(initial_peb)
            self.metrics['min_peb'].append(min_peb)
            self.metrics['rate'].append(avg_rate)
            self.metrics['power'].append(avg_power)
            self.metrics['reward'].append(avg_reward)
            
            if episode % 10 == 0:
                self.agent.save_models()
                print(f"\nEpisode {episode+1}/{num_episodes}")
                print(f"Initial PEB: {initial_peb}")
                print(f"Min PEB: {min_peb}")
                print(f"Average Rate: {avg_rate}")
                print(f"Average Power: {avg_power}")
                print(f"Average Reward: {avg_reward}")
                print("-"*50)
                
        return self.metrics
    
    def test(self, num_episodes=10):
        print("Testing...")
        for episode in range(num_episodes):
            state = np.array(self.eng.reset(self.sim))
            reward = 0
            done = False
            step = 0
            while not done and step<200:
                action = self.agent.choose_action(state, explore=False, epsilon=0)
                next_state, reward, done = self.eng.step(self.sim, action, nargout=3)
                state = np.array(next_state)
                step += 1
            print(f"Test episode {episode+1}/{num_episodes} completed with reward: {reward}")
        print("Testing completed.")
        
    def close(self):
        self.eng.quit()
        print("Training completed.")
    
if __name__ == "__main__":
    trainer = RISISACTrainer()
    metrics = trainer.train(num_episodes=300, max_steps=10000)
    trainer.test()
    trainer.close()
    
    plt_folder = 'tfddpg_plots'
    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)
        
    
    rewards = metrics['reward']
    episodes = list(range(len(rewards)))    
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, metrics['initial_peb'], label='Initial PEB')
    plt.xlabel('Episodes')
    plt.ylabel('PEB')
    plt.grid(True)
    plt.savefig(f'{plt_folder}/initial_peb.png')
    
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, metrics['min_peb'], label='Min PEB')
    plt.xlabel('Episodes')
    plt.ylabel('PEB')
    plt.grid(True)
    plt.savefig(f'{plt_folder}/min_peb.png')
    
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, metrics['rate'], label='Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Rate')
    plt.grid(True)
    plt.savefig(f'{plt_folder}/rate.png')
    
    plt.figure(figsize=(12, 8)) 
    plt.plot(episodes, metrics['power'], label='Power')
    plt.xlabel('Episodes')
    plt.ylabel('Power')
    plt.grid(True)
    plt.savefig(f'{plt_folder}/power.png')
    
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, rewards, label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.grid(True)
    plt.savefig(f'{plt_folder}/rewards.png')
    
    
            




