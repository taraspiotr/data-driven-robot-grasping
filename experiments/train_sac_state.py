from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

from grasp.sac.sac import sac
from grasp.sac.core import MLPActorCritic


def create_kuka_gym_env():
    return KukaGymEnv(renders=False, isDiscrete=False)


if __name__ == "__main__":
    seed = 0

    sac(
        env_fn=create_kuka_gym_env,
        seed=seed,
        actor_critic=MLPActorCritic,
        logger_kwargs={
            "output_dir": "/home/piotr/grasp/data/",
            "exp_name": "sac_state",
        },
    )
