import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

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
