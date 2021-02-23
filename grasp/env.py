import glob
import os
import numpy as np
from gym import spaces
from pybullet_envs.bullet.kuka_diverse_object_gym_env import (
    KukaDiverseObjectEnv as _KukaDiverseObjectEnv,
)
from rlpyt.envs.gym import GymEnvWrapper


class KukaDiverseObjectEnv(_KukaDiverseObjectEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.observation_space = spaces.Box(low=-8, high=8, shape=(4,))
        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=[self._height * self._width * 3]
        # )

    # def _get_random_object(self, num_objects, test):
    #     """Randomly choose an object urdf from the random_urdfs directory.
    #     Args:
    #       num_objects:
    #         Number of graspable objects.
    #     Returns:
    #       A list of urdf filenames.
    #     """
    #     if test:
    #         urdf_pattern = os.path.join(self._urdfRoot, "random_urdfs/*0/*.urdf")
    #     else:
    #         urdf_pattern = os.path.join(self._urdfRoot, "random_urdfs/*[1-9]/*.urdf")
    #     found_object_directories = glob.glob(urdf_pattern)
    #     # total_num_objects = len(found_object_directories)
    #     #         selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
    #     selected_objects = np.arange(num_objects)
    #     selected_objects_filenames = []
    #     for object_index in selected_objects:
    #         selected_objects_filenames += [found_object_directories[object_index]]
    #     return selected_objects_filenames

    def step(self, action):
        # print(action, type(action))
        self.time_step += 1
        if self._isDiscrete:
            o, r, d, i = super().step(int(action))
        else:
            o, r, d, i = super().step(action)
        # t = np.ones((self._height, self._width, 1)) * self.time_step
        # o = np.concatenate([o, t], axis=-1)
        # o = o.flatten() / 255.0
        # o = self.pos.copy()
        # print(o)
        # o[0] = self.time_step
        return o, r, d, i

    def _step_continuous(self, action):
        # print(action)
        self.pos += action[:4]
        return super()._step_continuous(action)

    def reset(self):
        # print(action, type(action))
        self.time_step = 0
        self.pos = np.zeros(4)
        o = super().reset()
        # t = np.ones((self._height, self._width, 1)) * self.time_step
        # o = np.concatenate([o, t], axis=-1)
        # o = o.flatten() / 255.0
        # o = self.pos.copy()
        # o[0] = self.time_step
        return o


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
