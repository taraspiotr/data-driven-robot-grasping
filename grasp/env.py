import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import (
    KukaDiverseObjectEnv as _KukaDiverseObjectEnv,
)
from gym import spaces
from rlpyt.envs.gym import GymEnvWrapper


class KukaDiverseObjectEnv(_KukaDiverseObjectEnv):

    def __init__(self, *args, reward_shaping: bool = False, from_pixels: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._reward_shaping = reward_shaping
        self._from_pixels = from_pixels

        if not self._from_pixels:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(4 + 7 * self._numObjects,))

    def _get_observation(self):
        if self._from_pixels:
            o = super()._get_observation()
            return o.transpose(2, 0, 1)
        else:
            o = self._kuka.endEffectorPos + [self._kuka.endEffectorAngle]
            for uid in self._objectUids:
                pos, quaternion = self._p.getBasePositionAndOrientation(uid)
                o += list(pos)
                o += list(quaternion)
            return np.array(o)

    def step(self, action):
        o, r, d, i = super().step(action)
        if self._reward_shaping and not self._isTest:
            grip_pos = np.array(self._kuka.endEffectorPos)
            obj_pos = np.array(self._p.getBasePositionAndOrientation(self._objectUids[0])[0])
            r = r * 10 - np.linalg.norm(obj_pos[:2] - grip_pos[:2])
        return o, r, d, i


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
    reward_shaping: bool = False,
    from_pixels: bool = True,
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
            reward_shaping=reward_shaping,
            from_pixels=from_pixels,
        )
    )
