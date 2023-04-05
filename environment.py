import numpy as np
import gym
from gym import spaces
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class EnergyArbitrageEnv(py_environment.PyEnvironment):

    def __init__(self, storage_capacity=10):
        super().__init__()
        self.storage_capacity = storage_capacity
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        self.energy_prices = np.array([])
        self.reset()

    def set_energy_prices(self, energy_prices):
        self.energy_prices = energy_prices

    def _reset(self):
        self.current_step = 0
        self.storage_level = 0
        return ts.restart(self._get_observation())

    def _step(self, action):
        if self.current_step == len(self.energy_prices) - 1:
            return ts.termination(self._get_observation(), 0)
        
        reward = 0
        price = self.energy_prices[self.current_step]
        next_price = self.energy_prices[self.current_step + 1] if self.current_step < len(self.energy_prices) - 2 else price
        
        if action == 0:  # Buy
            reward = next_price - price
            self.storage_level = min(self.storage_level + 1, self.storage_capacity)
        elif action == 1:  # Sell
            reward = price - next_price
            self.storage_level = max(self.storage_level - 1, 0)
        elif action == 2:  # Hold
            reward = 0

        self.current_step += 1

        return ts.transition(self._get_observation(), reward, discount=1.0)

    def _get_observation(self):
        if len(self.energy_prices) < 2:
            return np.zeros(4, dtype=np.float32)
        next_price = self.energy_prices[self.current_step + 1] if self.current_step < len(self.energy_prices) - 2 else self.energy_prices[self.current_step]
        return np.array([self.energy_prices[self.current_step], next_price, self.current_step, self.storage_level], dtype=np.float32)

    def action_spec(self):
        return array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

    def observation_spec(self):
        return array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=[0, 0, 0, 0], maximum=[np.inf, np.inf, np.inf, self.storage_capacity], name='observation')

    def _render(self, mode='human', close=False):
        pass
