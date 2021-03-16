from pybullet_envs.bullet.kuka_diverse_object_gym_env import (
    KukaDiverseObjectEnv as _KukaDiverseObjectEnv,
)
from rlpyt.envs.gym import GymEnvWrapper


class KukaDiverseObjectEnv(_KukaDiverseObjectEnv):
    def step(self, action):
        if self._isDiscrete:
            o, r, d, i = super().step(int(action))
        else:
            o, r, d, i = super().step(action)
        return o.transpose(2, 0, 1), r, d, i

    def reset(self):
        o = super().reset()
        return o.transpose(2, 0, 1)


def create_kuka_gym_diverse_env(
    is_discrete: bool = False,
    use_height_hack: bool = True,
    block_random: float = 0,
    camera_random: float = 0,
    test: bool = False,
    num_objects: int = 5,
    width: int = 64,
    height: int = 64,
    max_steps: int = 8,
):
    return GymEnvWrapper(
        KukaDiverseObjectEnv(
            renders=False,
            isDiscrete=is_discrete,
            removeHeightHack=not use_height_hack,
            blockRandom=block_random,
            cameraRandom=camera_random,
            numObjects=num_objects,
            isTest=test,
            width=width,
            height=height,
            maxSteps=max_steps,
        )
    )
