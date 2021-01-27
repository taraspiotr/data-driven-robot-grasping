from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from grasp.sac.sac import sac
from grasp.sac.core import CNNActorCritic


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
    )


if __name__ == "__main__":
    seed = 0

    sac(
        env_fn=create_kuka_gym_diverse_env,
        seed=seed,
        actor_critic=CNNActorCritic,
        logger_kwargs={
            "output_dir": "/home/piotr/grasp/data/",
            "exp_name": "sac_pixels",
        },
        replay_size=int(1e4),
    )
