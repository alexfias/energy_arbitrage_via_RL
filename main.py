import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from environment import EnergyArbitrageEnvironment

# Create energy prices data
energy_prices = np.random.uniform(10, 50, size=100)

# Environment parameters
storage_capacity = 10
storage_level = 0
storage_efficiency = 0.9

# Create the environment
env = EnergyArbitrageEnvironment(energy_prices, storage_capacity, storage_level, storage_efficiency)
train_env = tf_py_environment.TFPyEnvironment(env)

#
