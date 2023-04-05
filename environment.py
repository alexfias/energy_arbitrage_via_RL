import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class EnergyArbitrageEnvironment(py_environment.PyEnvironment):
    def __init__(self, energy_prices, storage_capacity, storage_level, storage_efficiency):
        self.energy_prices = energy_prices
        self.storage_capacity = storage_capacity
        self.storage_level = storage_level
        self.storage_efficiency = storage_efficiency
        self.current_step = 0

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=[0, 0, 0, 0], maximum=[np.finfo(np.float32).max, np.finfo(np.float32).max, len(energy_prices)-1, storage_capacity], name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.current_step = 0
        self.storage_level = 0
        return ts.restart(np.array([self.energy_prices[0], self.energy_prices[1], 0, self.storage_level], dtype=np.float32))

    def _step(self, action):
        if self.current_step == len(self.energy_prices) - 2:
            return self.reset()

        reward = 0
        price = self.energy_prices[self.current_step]
        next_price = self.energy_prices[self.current_step + 1]

        if action == 0:  # Buy
            energy_to_buy = min(self.storage_capacity - self.storage_level, self.storage_level / price)
            self.storage_level += energy_to_buy * self.storage_efficiency
            reward = -energy_to_buy * price
        elif action == 1:  # Sell
            energy_to_sell = min(self.storage_level, self.storage_level * price)
            self.storage_level -= energy_to_sell
            reward = energy_to_sell * next_price

        self.current_step += 1
        next_observation = np.array([self.energy_prices[self.current_step], self.energy_prices[self.current_step + 1], self.current_step, self.storage_level], dtype=np.float32)
        return ts.transition(next_observation, reward)
