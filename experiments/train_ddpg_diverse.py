# from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from grasp.ddpg.core import MLPActorCritic, CNNActorCritic
from grasp.ddpg.ddpg import ddpg
from grasp.env import KukaDiverseObjectEnv


def create_kuka_gym_diverse_env():
    return KukaDiverseObjectEnv(
        renders=False,
        isDiscrete=False,
        blockRandom=0,
        cameraRandom=0,
        numObjects=1,
        isTest=False,
        width=64,
        height=64,
        maxSteps=10,
    )


if __name__ == "__main__":
    seed = 0

    # for alpha in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 10, 100):
    ddpg(
        env_fn=create_kuka_gym_diverse_env,
        seed=seed,
        actor_critic=CNNActorCritic,
        replay_size=int(1e5),
    )
