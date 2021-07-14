from typing import Sequence

import numpy as np
import torch
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from grasp.encoder import make_encoder


class QofMuMlpModel(torch.nn.Module):
    """Q portion of the model for DDPG, an MLP."""

    def __init__(
        self, observation_shape, hidden_sizes, action_size,
    ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        observation_shape = observation_shape["state"]
        self._obs_ndim = len(observation_shape)
        print("value function", observation_shape)
        self.mlp1 = MlpModel(
            input_size=int(np.prod(observation_shape)) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.mlp2 = MlpModel(
            input_size=int(np.prod(observation_shape)) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action):
        observation = observation.state
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1
        )
        q1 = self.mlp1(q_input).squeeze(-1)
        q2 = self.mlp2(q_input).squeeze(-1)
        q1, q2 = restore_leading_dims(q1, lead_dim, T, B), restore_leading_dims(q2, lead_dim, T, B)
        return q1, q2



class QofMuCnnModel(torch.nn.Module):
    """Q portion of the model for DDPG, an MLP."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_sizes: Sequence[int],
        encoder_feature_dim: int = 32,
        encoder_num_layers: int = 2,
        encoder_num_filters: int = 32,
    ):
        """Instantiate neural net according to inputs."""
        observation_shape = observation_shape["pixels"]
        super().__init__()
        self._obs_ndim = len(observation_shape)
        print("value function", observation_shape)

        self.encoder = make_encoder(
            "pixel",
            obs_shape=observation_shape,
            feature_dim=encoder_feature_dim,
            num_layers=encoder_num_layers,
            num_filters=encoder_num_filters,
        )
        self.q1_head = MlpModel(
            input_size=encoder_feature_dim + action_size,
            hidden_sizes=list(hidden_sizes),
            output_size=1,
        )
        self.q2_head = MlpModel(
            input_size=encoder_feature_dim + action_size,
            hidden_sizes=list(hidden_sizes),
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action):
        observation = observation.pixels
        lead_dim, T, B, shape = infer_leading_dims(observation, self._obs_ndim)
        encoder_output = self.encoder(observation.view(T * B, *shape))
        q_input = torch.cat([encoder_output, action.view(T * B, -1)], dim=1)
        q1 = self.q1_head(q_input).squeeze(-1)
        q2 = self.q2_head(q_input).squeeze(-1)
        q1, q1 = restore_leading_dims(q1, lead_dim, T, B), restore_leading_dims(q2, lead_dim, T, B)
        return q1, q2
