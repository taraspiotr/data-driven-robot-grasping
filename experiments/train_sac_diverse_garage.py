# from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from grasp.sac.sac import sac
from grasp.sac.core import CNNActorCritic, MLPActorCritic
from grasp.env import KukaDiverseObjectEnv


def create_kuka_gym_diverse_env():
    return KukaDiverseObjectEnv(
        renders=False,
        isDiscrete=False,
        blockRandom=0,
        cameraRandom=0,
        numObjects=1,
        isTest=False,
        width=32,
        height=32,
        maxSteps=10,
    )


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import normalize, GarageEnv
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage import wrap_experiment
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.replay_buffer import PathBuffer


@wrap_experiment(snapshot_mode="none")
def sac_half_cheetah_batch(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(create_kuka_gym_diverse_env())

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.0),
        max_std=np.exp(2.0),
    )

    qf1 = ContinuousMLPQFunction(
        env_spec=env.spec, hidden_sizes=[256, 256], hidden_nonlinearity=F.relu
    )

    qf2 = ContinuousMLPQFunction(
        env_spec=env.spec, hidden_sizes=[256, 256], hidden_nonlinearity=F.relu
    )

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    # sampler = LocalSampler(
    #     agents=policy, envs=env, max_episode_length=10, worker_class=DefaultWorker,
    # )

    sac = SAC(
        env_spec=env.spec,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        gradient_steps_per_itr=1000,
        max_path_length=500,
        replay_buffer=replay_buffer,
        min_buffer_size=1e4,
        target_update_tau=5e-3,
        discount=0.99,
        buffer_batch_size=256,
        reward_scale=1.0,
        steps_per_epoch=1,
    )

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    runner.setup(
        algo=sac, env=env, sampler_cls=LocalSampler, worker_class=DefaultWorker
    )
    runner.train(n_epochs=1000, batch_size=1000)


s = np.random.randint(0, 1000)
sac_half_cheetah_batch(seed=521)