# RL Pong Agent

This project implements a Deep Q-Network (DQN) agent to play Atari Pong using PyTorch and Gymnasium. The codebase includes all components needed for training, evaluating, and saving a reinforcement learning agent.

## Project Structure

```
.
├── agent.py         # Main agent logic (training loop, action selection, etc.)
├── buffer.py        # Replay buffer for experience replay
├── model.py         # Neural network model for Q-learning
├── requirements.txt # Python dependencies
├── test.py          # (Not detailed here; assumed for testing)
├── train.py         # Script to train the agent
├── .gitignore       # Files/folders to ignore in git
├── .vscode/         # VSCode configuration
└── __pycache__/     # Python bytecode cache
```

---

## Main Components

### [`train.py`](train.py)
- Sets up the Pong environment using Gymnasium.
- Applies wrappers for resizing, grayscale conversion, and frame stacking.
- Initializes the [`Agent`](agent.py) and starts the training loop.

### [`agent.py`](agent.py)
- Defines the `Agent` class, which:
  - Handles environment interaction, action selection (epsilon-greedy), and training.
  - Uses a [`ReplayBuffer`](buffer.py) for experience replay.
  - Maintains both a main and target Q-network ([`Model`](model.py)).
  - Logs training progress with TensorBoard.
  - Periodically saves the model.

### [`buffer.py`](buffer.py)
- Implements the `ReplayBuffer` class:
  - Stores transitions (state, action, reward, next_state, done).
  - Supports sampling random minibatches for training.
  - Handles memory management and device placement.

### [`model.py`](model.py)
- Defines the `Model` class:
  - A convolutional neural network for estimating Q-values.
  - Includes methods for saving/loading model weights.
- Provides a `soft_update` function for Polyak averaging of target network parameters.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd rl_pong_agent
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

To train the agent, run:

```sh
python train.py
```

Training logs will be saved for TensorBoard visualization in the `runs/` directory. Model checkpoints are saved in the `models/` directory.

---

## Key Features

- **Replay Buffer:** Efficient experience replay for stable training ([`ReplayBuffer`](buffer.py)).
- **Deep Q-Network:** Convolutional neural network for processing stacked grayscale frames ([`Model`](model.py)).
- **Target Network:** Stabilizes learning by slowly updating target Q-network weights.
- **Frame Stacking & Preprocessing:** Handles Atari-specific preprocessing (resize, grayscale, stack).
- **TensorBoard Logging:** Tracks loss, rewards, and epsilon over time.

---

## Customization

- **Hyperparameters:** Adjust in [`train.py`](train.py) (e.g., learning rate, batch size, epsilon decay).
- **Model Architecture:** Modify in [`model.py`](model.py).
- **Environment:** Swap out Pong for other Atari games by changing the environment name in [`train.py`](train.py).

---

## Notes

- The `.gitignore` excludes model checkpoints, TensorBoard logs, and Python bytecode.
- The code is designed for easy extension and experimentation with different RL algorithms or environments.

---

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies, including:
- `gymnasium[atari]`
- `ale-py`
- `torch`
- `opencv-python`
- `tensorboard`
- `pygame`
- `torchvision`, `torchaudio`

---

## License

MIT License (add your own license if needed).

---

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/)