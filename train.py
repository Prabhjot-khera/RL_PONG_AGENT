import gymnasium as gym
import numpy as np
from collections import deque
from agent import Agent
from gymnasium.wrappers import ResizeObservation, TransformObservation
import ale_py

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        obs_space = env.observation_space
        low = np.repeat(obs_space.low, k, axis=-1)
        high = np.repeat(obs_space.high, k, axis=-1)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=obs_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

def to_grayscale(obs):
    return np.mean(obs, axis=-1, keepdims=True).astype(np.uint8)

hidden_layer = 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99
episodes = 10000
max_episode_steps = 10000
batch_size = 64
epsilopn = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")


env = ResizeObservation(env, (64, 64))
env = TransformObservation(env, to_grayscale, observation_space=env.observation_space)
env = FrameStack(env, 4)

agent = Agent(
    env,
    hidden_layer=hidden_layer,
    learning_rate=learning_rate,
    step_repeat=step_repeat,
    gamma=gamma
)

summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

agent.train(
    episodes=episodes,
    max_episode_steps=max_episode_steps,
    summary_writer_suffix=summary_writer_suffix,
    batch_size=batch_size,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon
)

