from rlpyt.agents.qpg.sac_agent import SacAgent

import grasp.data_augs as rad


class RadSacAgent(SacAgent):
    def __init__(self, *args, augmentations, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = augmentations

    def step(self, observation, prev_action, prev_reward):
        if "crop" in self.augmentations:
            observation = rad.center_crop(observation)
        return super().step(observation, prev_action, prev_reward)
