from collections import OrderedDict

import numpy as np
import scipy.signal

import torch
import torch.nn as nn

from grasp.encoder import make_encoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class CNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        feature_dim = 32
        encoder = make_encoder(
            "pixel",
            obs_shape=obs_dim,
            feature_dim=feature_dim,
            num_layers=2,
            num_filters=32,
        )
        head = mlp([feature_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh)
        self.pi = nn.Sequential(OrderedDict([("encoder", encoder), ("head", head),]))
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class CNNQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        feature_dim = 32
        encoder = make_encoder(
            "pixel",
            obs_shape=obs_dim,
            feature_dim=feature_dim,
            num_layers=2,
            num_filters=32,
        )
        head = mlp([feature_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.q = nn.Sequential(OrderedDict([("encoder", encoder), ("head", head),]))

    def forward(self, obs, act):
        enc = self.q.encoder(obs)
        q = self.q.head(torch.cat([enc, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class CNNActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = CNNActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = CNNQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs.to(DEVICE).unsqueeze(0)).cpu().squeeze(0).numpy()
