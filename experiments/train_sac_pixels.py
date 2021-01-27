from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv

from grasp.sac.sac import sac
from grasp.sac.core import CNNActorCritic


def create_kuka_gym_cam_env():
    return KukaCamGymEnv(renders=False, isDiscrete=False)


if __name__ == "__main__":
    seed = 0

    sac(
        env_fn=create_kuka_gym_cam_env,
        seed=seed,
        actor_critic=CNNActorCritic,
        logger_kwargs={
            "output_dir": "/home/piotr/grasp/data/",
            "exp_name": "sac_pixels",
        },
        replay_size=int(1e4),
    )
