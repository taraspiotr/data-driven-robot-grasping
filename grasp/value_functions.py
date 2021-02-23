from typing import Sequence

import torch
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from grasp.encoder import make_encoder


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
        super().__init__()
        self._obs_ndim = len(observation_shape)

        self.encoder = make_encoder(
            "pixel",
            obs_shape=observation_shape,
            feature_dim=encoder_feature_dim,
            num_layers=encoder_num_layers,
            num_filters=encoder_num_filters,
        )
        self.head = MlpModel(
            input_size=encoder_feature_dim + action_size,
            hidden_sizes=list(hidden_sizes),
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action):
        lead_dim, T, B, shape = infer_leading_dims(observation, self._obs_ndim)
        encoder_output = self.encoder(observation.view(T * B, *shape))
        q_input = torch.cat([encoder_output, action.view(T * B, -1)], dim=1)
        q = self.head(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q
