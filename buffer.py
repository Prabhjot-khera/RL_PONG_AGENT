import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.device = device

