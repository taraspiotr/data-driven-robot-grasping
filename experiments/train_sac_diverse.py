# from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import torch

from grasp.sac.sac import sac
from grasp.sac.core import CNNActorCritic, MLPActorCritic
from grasp.env import KukaDiverseObjectEnv


def create_kuka_gym_diverse_env(test: bool = False):
    return KukaDiverseObjectEnv(
        renders=False,
        isDiscrete=False,
        removeHeightHack=False,
        blockRandom=0,
        cameraRandom=0,
        numObjects=5,
        isTest=test,
        width=64,
        height=64,
        maxSteps=15,
    )


if __name__ == "__main__":
    print(f"CUDA IS{'' if torch.cuda.is_available() else ' NOT'} AVAILABLE!")

    seed = 0

    # for alpha in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 10, 100):
    sac(
        env_fn=create_kuka_gym_diverse_env,
        seed=seed,
        actor_critic=CNNActorCritic,
        logger_kwargs={"output_dir": "/home/ucabpmt/data/", "exp_name": "sac_pixels",},
        replay_size=int(1e5),
        steps_per_epoch=1000,
        epochs=1000,
        batch_size=100,
        start_steps=1000,
        update_after=100,
        update_every=10,
        num_test_episodes=10,
        max_ep_len=100,
        save_freq=1,
        gamma=0.95,
        # ac_kwargs={"hidden_sizes": [16, 16]},
        alpha=1e-3,
        lr=1e-3,
    )
