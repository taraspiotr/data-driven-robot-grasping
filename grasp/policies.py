from typing import Sequence, Dict

import numpy as np
import torch
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from grasp.encoder import make_encoder


class PiMlpModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
        self, observation_shape, hidden_sizes, action_size,
    ):
        observation_shape = observation_shape["state"]
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        print("policy", observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward(self, observation, prev_action, prev_reward):
        observation = observation.state
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = output[:, : self._action_size], output[:, self._action_size :]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class PiCnnModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_sizes: Sequence[int],
        encoder_feature_dim: int = 32,
        encoder_num_layers: int = 2,
        encoder_num_filters: int = 32,
        detach_encoder: bool = False
    ):
        super().__init__()
        observation_shape = observation_shape["pixels"]
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        print("policy", observation_shape)

        self.encoder = make_encoder(
            "pixel",
            obs_shape=observation_shape,
            feature_dim=encoder_feature_dim,
            num_layers=encoder_num_layers,
            num_filters=encoder_num_filters,
        )
        self.head = MlpModel(
            input_size=encoder_feature_dim,
            hidden_sizes=list(hidden_sizes),
            output_size=action_size * 2,
        )
        self._detach_encoder = detach_encoder


    def forward(
        self,
        observation: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
    ):
        observation = observation.pixels
        lead_dim, T, B, shape = infer_leading_dims(observation, self._obs_ndim)
        
        encoder_output = self.encoder(observation.view(T * B, *shape), detach=self._detach_encoder)
        output = self.head(encoder_output)
        mu, log_std = output[:, : self._action_size], output[:, self._action_size :]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std
