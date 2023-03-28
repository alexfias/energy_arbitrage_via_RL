from environment import EnergyArbitrageEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

#load energy price data
energy_prices = np.array([...])

storage_capacity = 10
env = EnergyArbitrageEnv(storage_capacity=storage_capacity)
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
