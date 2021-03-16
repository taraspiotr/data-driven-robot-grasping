from typing import Dict, Any

import functools

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.qpg.sac import SAC

# from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from mrunner.helpers.client_helper import get_configuration

from grasp.env import create_kuka_gym_diverse_env
from grasp.logger import setup_logger
from grasp.policies import PiCnnModel
from grasp.value_functions import QofMuCnnModel
from grasp.collectors import SerialEvalCollectorLogger

# from grasp.sampler import RadSerialSampler
from grasp.data_augs import get_augmentations
from grasp.agent import RadSacAgent
from grasp.replay import UniformRadReplayBuffer


def build_and_train(config: Dict[str, Any]):

    env_kwargs: Dict[str, Any] = dict(
        num_objects=config["env_num_objects"],
        camera_random=config["env_camera_random"],
        block_random=config["env_block_random"],
        use_height_hack=config["env_use_height_hack"],
        width=config.get("observation_size", 64),
        height=config.get("observation_size", 64),
    )
    augmentations = config.get("augmentations", [])
    sampler = SerialSampler(
        EnvCls=create_kuka_gym_diverse_env,
        env_kwargs=dict(test=False, **env_kwargs),
        eval_env_kwargs=dict(test=True, **env_kwargs),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
        eval_CollectorCls=SerialEvalCollectorLogger,
    )
    algo = SAC(
        bootstrap_timelimit=False,
        min_steps_learn=int(1e3),
        fixed_alpha=config["alpha"],
        learning_rate=config["learning_rate"],
        ReplayBufferCls=functools.partial(
            UniformRadReplayBuffer, augs_func=get_augmentations(augmentations)
        ),
    )  # Run with defaults.

    model_kwargs: Dict[str, Any] = dict(
        hidden_sizes=config["model_hidden_sizes"],
        encoder_feature_dim=config["encoder_feature_dim"],
        encoder_num_layers=config["encoder_num_layers"],
        encoder_num_filters=config["encoder_num_filters"],
    )
    agent = RadSacAgent(
        ModelCls=PiCnnModel,
        QModelCls=QofMuCnnModel,
        model_kwargs=model_kwargs,
        q_model_kwargs=model_kwargs,
        augmentations=augmentations,
    )
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=config.get("cuda_idx")),
        seed=config.get("seed"),
    )
    log_dir = f"/tmp/{config['name']}"
    with logger_context(log_dir, 0, config["name"], config, override_prefix=True):
        runner.train()


if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=True)
    setup_logger()
    experiment_id = config.pop("experiment_id")
    build_and_train(config)
