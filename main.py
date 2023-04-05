import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from environment import EnergyArbitrageEnv

# Replace with your own energy price data
energy_prices = np.array([10, 20, 15, 12, 8, 5, 30, 25, 18, 22, 11, 14, 9, 6, 31, 26,10, 20, 15, 12, 8, 5, 30, 25, 18, 22, 11, 14, 9, 6, 31, 26])

storage_capacity = 10
env = EnergyArbitrageEnv(storage_capacity=storage_capacity)
env.set_energy_prices(energy_prices)

train_py_env = EnergyArbitrageEnv(storage_capacity=storage_capacity)
train_py_env.set_energy_prices(energy_prices)
eval_py_env = EnergyArbitrageEnv(storage_capacity=storage_capacity)
eval_py_env.set_energy_prices(energy_prices)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (32, 32)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

global_step = tf.compat.v1.train.get_or_create_global_step()

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)
agent.initialize()

replay_buffer_capacity = 10000

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

num_episodes = 100

for episode in range(num_episodes):
    time_step = train_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action_step = agent.collect_policy.action(time_step)
        next_time_step = train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        replay_buffer.add_batch(traj)

        episode_return += time_step.reward
        time_step = next_time_step

    print(f"Episode {episode}: {episode_return.numpy()}")
