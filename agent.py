from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
import torch.utils.tensorboard as SummaryWriter
import random
import os
import cv2

class Agent():
    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):
        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma

        obs, info = self.env.reset()

        # TODO - Process Observation

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f'Loaded model on device {self.device}')
