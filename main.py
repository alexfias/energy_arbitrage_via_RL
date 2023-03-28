import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


#define the environment

class EnergyArbitrageEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(13,), dtype=np.float32)
        self.energy_prices = np.array([])
        self.current_step = 0
        self.storage_level = 0

    def set_energy_prices(self, prices):
        self.energy_prices = prices

    def step(self, action):
        if action == 0:  # Buy
            self.storage_level += 1
        elif action == 1:  # Sell
            self.storage_level = max(0, self.storage_level - 1)

        self.current_step += 1

        done = self.current_step >= len(self.energy_prices) - 12
        reward = -self.energy_prices[self.current_step - 1] if action == 0 else self.energy_prices[self.current_step - 1] if action == 1 else 0

        state = np.concatenate(([self.storage_level], self.energy_prices[self.current_step:self.current_step + 12]))
        return state, reward, done, {}

    def reset(self):
        self.storage_level = 0
        self.current_step = 0
        state = np.concatenate(([self.storage_level], self.energy_prices[self.current_step:self.current_step + 12]))
        return state

energy_prices = np.array([...])

env = EnergyArbitrageEnv()
env.set_energy_prices(energy_prices)

hidden_units = 32
action_size = env.action_space.n
memory_size = 10000
nb_steps_warmup = 1000
target_model_update = 1e-2

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dense(action_size))
model.add(Activation('linear'))

memory = SequentialMemory(limit=memory_size, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=nb_steps_warmup,
               target_model_update=target_model_update, policy=policy)
dqn.compile(Adam(lr=0.001), metrics=['mae'])

#train and test the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

dqn.test(env, nb_episodes=100, visualize=False)
