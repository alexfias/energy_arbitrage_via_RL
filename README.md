# energy_arbitrage_via_RL
Reinforcement learning model for energy arbitrage of a storage


# Energy Arbitrage with Reinforcement Learning

This project uses reinforcement learning to optimize energy arbitrage in an energy market. It is based on Keras-RL and a custom energy market environment built using OpenAI Gym.

## Requirements

- Python 3.7+
- Keras
- Keras-RL
- OpenAI Gym
- NumPy

You can install the required packages using the following command:

pip install keras keras-rl2 gym numpy


## Usage

1. Clone the repository:

git clone https://github.com/username/energy-arbitrage.git
cd energy-arbitrage


2. Add your own energy price data to the `energy_prices` variable in the `main.py` script:

energy_prices = np.array(...)  # Replace with your own data

3. Train the reinforcement learning agent

python main.py --mode train

4. Test the trained agent:

python main.py --mode test
