import gymnasium as gym
import numpy as np
import torch

from sac import PolicyNetwork


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.actor = PolicyNetwork(3, self.action_space.shape[0], self.action_space.high)
        self.actor.load_state_dict(torch.load("../models/Q1/checkpoint-600/actor.pt"))

    @torch.no_grad()
    def act(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        return self.actor(observation, add_noise=False).numpy()
