import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

class QNetworks(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)

    def forward(self, x):
        return self.q1(x), self.q2(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_max_values: np.ndarray):
        """
        output action_i in (-action_max_values_i, action_max_values_i)
        """
        super().__init__()
        self.register_buffer("action_max_values", torch.FloatTensor(action_max_values))

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        # learn log_sigma and take exp() when needed to ensure that sigma > 0
        self.log_sigma = nn.Linear(256, action_dim)

    def forward(self, x, return_log_pi=False, add_noise=True):
        """
        :return action: of shape (batch_size, action_dim)
        :return log_pi (optional): of shape (batch_size,)
        """
        x = self.backbone(x)
        mu = self.mu(x)

        if not add_noise:
            return mu.tanh() * self.action_max_values

        sigma = self.log_sigma(x).clamp(-20, 2).exp()
        distribution = Normal(mu, sigma)
        sample = distribution.rsample()

        if return_log_pi:
            log_pi = distribution.log_prob(sample).sum(-1)\
                    - (2 * (np.log(2) - sample - F.softplus(-2 * sample))).sum(-1)  # numerically stable version of tanh() correction term

        action = sample.tanh() * self.action_max_values
        return (action, log_pi) if return_log_pi else action

class SAC:
    def __init__(self, state_dim, action_dim, action_max_values: np.ndarray,
                 args, checkpoint_dir=None):
        self.device = args.device
        self.gamma = args.gamma
        self.tau = args.tau

        self.actor = PolicyNetwork(state_dim, action_dim, action_max_values)
        self.critic = QNetworks(state_dim, action_dim)

        self.target_critic = QNetworks(state_dim, action_dim)
        for param in self.target_critic.parameters():
            param.requires_grad = False

        if checkpoint_dir is None:
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            self.actor.load_state_dict(torch.load(
                checkpoint_dir / "actor.pt", weights_only=False, map_location=self.device
            ))

            training_state = torch.load(checkpoint_dir / "training_state.pt", weights_only=False)
            self.critic.load_state_dict(training_state["critic"])
            self.target_critic.load_state_dict(training_state["target_critic"])

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

        if checkpoint_dir is None:
            # learn log_alpha and take exp() when needed to ensure that alpha > 0
            self.log_alpha = torch.tensor([np.log(args.alpha_init)], dtype=torch.float32,
                                          requires_grad=True, device=self.device)
        else:
            self.log_alpha = training_state["log_alpha"]
        self.entropy0 = -action_dim

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=args.learning_rate)
        self.optim_alpha = optim.Adam([self.log_alpha], lr=args.learning_rate)

        if checkpoint_dir is not None:
            self.optim_actor.load_state_dict(training_state["optim_actor"])
            self.optim_critic.load_state_dict(training_state["optim_critic"])
            self.optim_alpha.load_state_dict(training_state["optim_alpha"])

        self.criterion_critic = nn.MSELoss()

    @torch.no_grad()
    def get_action(self, state: np.ndarray):
        state = torch.FloatTensor(state, device=self.device)
        return self.actor(state).cpu().numpy()

    def soft_update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_online(self, states, actions, rewards, next_states, dones):
        """
        :param: All the arguments should be tensors on args.device.
        """
        # update critic
        alpha_attached = self.log_alpha.exp()
        alpha = alpha_attached.detach()

        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_states, return_log_pi=True)
            target_q1, target_q2 = self.target_critic(torch.cat((next_states, next_actions), 1))
            target = rewards + (1 - dones) * self.gamma * (
                torch.min(target_q1, target_q2) - alpha * next_log_pi
            )

        q1, q2 = self.critic(torch.cat((states, actions), 1))
        loss_critic = self.criterion_critic(q1, target) + self.criterion_critic(q2, target)

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()


        # update actor
        for param in self.critic.parameters():
            param.requires_grad = False

        current_actions, current_log_pi = self.actor(states, return_log_pi=True)
        q1, q2 = self.critic(torch.cat((states, current_actions), 1))
        loss_actor = (alpha * current_log_pi - torch.min(q1, q2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        for param in self.critic.parameters():
            param.requires_grad = True


        # update alpha
        loss_alpha = -alpha_attached * (current_log_pi.detach().mean() + self.entropy0)

        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()


        # update target critic
        self.soft_update_target()

    def save(self, output_dir):
        self.actor.cpu()
        torch.save(self.actor.state_dict(), output_dir / "actor.pt")
        self.actor.to(self.device)

        torch.save({
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha,
            "optim_actor": self.optim_actor.state_dict(),
            "optim_critic": self.optim_critic.state_dict(),
            "optim_alpha": self.optim_alpha.state_dict(),
        }, output_dir / "training_state.pt")

    @torch.no_grad()
    def log(self):
        logger.info(f"Alpha: {self.log_alpha.exp().item()}")
