# Energy Arbitrage Reinforcement Learning

This repository contains a reinforcement learning (RL) model to optimize energy arbitrage in a storage system. The model is built using the TensorFlow Agents library and is designed to help an agent make decisions on when to buy, sell, or hold energy based on the energy prices.

## Environment

The environment simulates an energy market with a storage system. The agent interacts with the environment by choosing actions from a discrete action space:

0. Buy energy
1. Sell energy
2. Hold (do nothing)

The agent receives a reward based on the difference between the energy prices when buying and selling, as well as the storage level and efficiency.

## Model

The RL agent used in this model is a DQN (Deep Q-Network) agent, which employs a neural network to approximate the Q-function. The Q-function represents the expected cumulative reward the agent can obtain by taking a specific action in a given state and following the optimal policy thereafter. The DQN agent learns from experience by interacting with the environment and optimizing the neural network to minimize the difference between the predicted and actual Q-values.

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.5+
- TensorFlow Agents

### Running the Model

1. Clone the repository:


git clone https://github.com/your_username/energy-arbitrage-rl.git

2. Install the required packages:


pip install tensorflow tensorflow-agents

3. Update the `energy_prices` variable in `main.py` with your own energy price data.

4. Run the `main.py` file to train the RL agent:

python main.py


4. Observe the training progress and the performance of the trained agent during testing.

## Customization

You can customize the storage capacity of the environment by modifying the `storage_capacity` variable in `main.py`. The default storage capacity is set to 10.

To customize the DQN model architecture or training parameters, you can modify the respective sections in `main.py`.



