import gymnasium
import numpy as np
import torch

from sac import PolicyNetwork


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.actor = PolicyNetwork(5, self.action_space.shape[0], self.action_space.high)
        self.actor.load_state_dict(torch.load("../models/Q2/checkpoint-600/actor.pt"))

    @torch.no_grad()
    def act(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        return self.actor(observation, add_noise=False).numpy()
