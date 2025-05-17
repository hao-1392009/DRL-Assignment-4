import random
import collections

import torch


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            torch.FloatTensor(state, device=self.device),
            torch.FloatTensor(action, device=self.device),
            torch.FloatTensor([reward], device=self.device),
            torch.FloatTensor(next_state, device=self.device),
            torch.FloatTensor([done], device=self.device)
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.stack(states, 0),  # (batch_size, state_dim)
            torch.stack(actions, 0),  # (batch_size, action_dim)
            torch.cat(rewards, 0),  # (batch_size,)
            torch.stack(next_states, 0),  # (batch_size, state_dim)
            torch.cat(dones, 0)  # (batch_size,)
        )

    def __len__(self):
        return len(self.buffer)
