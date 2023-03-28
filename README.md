# Energy Arbitrage Reinforcement Learning

This repository contains a simple reinforcement learning model for energy arbitrage in a storage system. The model is trained using Deep Q-Networks (DQN) to learn when to buy, sell or hold energy based on given energy price data.

## Files

- `environment.py`: Contains the custom `EnergyArbitrageEnv` class, which extends OpenAI Gym's environment class. This file defines the energy market environment and its dynamics.

- `main.py`: The main script that trains and tests the DQN agent using the `EnergyArbitrageEnv` environment.

## Dependencies

To run the code, you will need the following libraries:

- `gym`
- `numpy`
- `keras`
- `keras-rl2`

You can install these dependencies using pip:

pip install gym numpy keras keras-rl2


## Usage

1. Clone the repository:


git clone https://github.com/your_username/energy-arbitrage-rl.git
cd energy-arbitrage-rl


2. Update the `energy_prices` variable in `main.py` with your own energy price data.

3. Run the main script to train and test the model:

python main.py


4. Observe the training progress and the performance of the trained agent during testing.

## Customization

You can customize the storage capacity of the environment by modifying the `storage_capacity` variable in `main.py`. The default storage capacity is set to 10.

To customize the DQN model architecture or training parameters, you can modify the respective sections in `main.py`.



