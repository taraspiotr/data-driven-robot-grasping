import os
import pickle
import random

import numpy as np
from mrunner.helpers.client_helper import get_configuration

from grasp.env import create_kuka_gym_diverse_env


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def sample_trajectory(env, agent):
    o, r, d, _ = env.reset(), 0, False, None
    obs = [o]
    rewards = [r]
    actions = []
    while not d:
        a = agent.act(o, r, d)
        o, r, d, _ = env.step(a)
        obs.append(o)
        rewards.append(r)
        actions.append(a)
    return np.array(obs), np.array(rewards), np.array(actions)


def main(num_traj, storage_dir, is_discrete, **kwargs):
    env = create_kuka_gym_diverse_env(is_discrete=is_discrete)
    agent = RandomAgent(env.action_space)
    trajectories = []

    for _ in range(num_traj):
        trajectories.append(sample_trajectory(env, agent))

    id = "".join(random.choice("0123456789abcdef") for i in range(16))
    with open(f"traj_{id}.pkl", "wb") as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=False)
    main(**config)
