from rlpyt.agents.qpg.sac_agent import SacAgent as _SacAgent

import grasp.data_augs as rad


class SacAgent(_SacAgent):
    def __init__(self, *args, augmentations, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = augmentations

    # def initialize(self, env_spaces, share_memory, global_B, env_ranks):
    #     ret = super().initialize(env_spaces, share_memory=share_memory, global_B=global_B, env_ranks=env_ranks)
    #     self.q1_model.encoder.copy_conv_weights_from(self.model.encoder)
    #     self.q2_model.encoder.copy_conv_weights_from(self.model.encoder)
    #     return ret
        

    # def step(self, observation, prev_action, prev_reward):
    #     if "crop" in self.augmentations:
    #         observation = rad.center_crop(observation)
    #     return super().step(observation, prev_action, prev_reward)

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape={k: s.shape for k, s in env_spaces.observation._gym_space.spaces.items()},
            action_size=env_spaces.action.shape[0],
        )

    # def pi_parameters(self):
    #     return [p for n, p in self.model.named_parameters() if "conv" not in n]

    # def q1_parameters(self):
    #     return [p for n, p in self.q1_model.named_parameters() if "conv" not in n]

    # def q2_parameters(self):
    #     return [p for n, p in self.q2_model.named_parameters() if "conv" not in n]

    # def encoder_parameters(self):
    #     return [p for n, p in self.model.encoder.named_parameters() if "conv" in n]