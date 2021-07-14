from typing import Dict, Any

import functools

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
# from rlpyt.algos.qpg.sac import SAC
from rlpyt.utils.launching.affinity import (
    make_affinity,
    encode_affinity,
    decode_affinity,
    remove_run_slot,
)
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
# from rlpyt.models.qpg.mlp import PiMlpModel, QofMuMlpModel
from mrunner.helpers.client_helper import get_configuration

from grasp.env import create_kuka_gym_diverse_env
from grasp.logger import setup_logger
from grasp.policies import PiCnnModel, PiMlpModel
from grasp.value_functions import QofMuCnnModel, QofMuMlpModel

from grasp.collectors import SerialEvalCollectorLogger
from grasp.data_augs import get_augmentations
from grasp.agent import SacAgent
from grasp.replay import UniformRadReplayBuffer
from grasp.sac import SAC


def build_and_train(config: Dict[str, Any]):

    from_state = config["from_state"] or config["aac"]
    from_pixels = config["from_pixels"] or config["aac"]
    augmentations = config["augmentations"]

    env_kwargs: Dict[str, Any] = dict(
        num_objects=config["env_num_objects"],
        camera_random=config["env_camera_random"],
        block_random=config["env_block_random"],
        use_height_hack=config["env_use_height_hack"],
        width=config["observation_size"],
        height=config["observation_size"],
        reward_shaping=config["reward_shaping"],
        from_state=from_state,
        from_pixels=from_pixels,
    )

    # sampler = CpuSampler(
    #     EnvCls=create_kuka_gym_diverse_env,
    #     env_kwargs=dict(test=False, **env_kwargs),
    #     eval_env_kwargs=dict(test=True, **env_kwargs),
    #     batch_T=1,  # One time-step per sampler iteration.
    #     batch_B=config.get("batch_B", 1),
    #     max_decorrelation_steps=0,
    #     eval_n_envs=config.get("eval_n_envs", 1),
    #     eval_max_steps=int(51e3),
    #     eval_max_trajectories=50,
    # )
    
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
        lambda_=config["lambda"]
    )  # Run with defaults.


    if config["aac"]:
        pi_model = PiCnnModel
        pi_model_kwargs: Dict[str, Any] = dict(
            hidden_sizes=config["model_hidden_sizes"],
            encoder_feature_dim=config["encoder_feature_dim"],
            encoder_num_layers=config["encoder_num_layers"],
            encoder_num_filters=config["encoder_num_filters"],
        )
        q_model = QofMuMlpModel
        q_model_kwargs = dict(
            hidden_sizes=config["model_hidden_sizes"],
        )
    else:
        pi_model = PiCnnModel
        pi_model_kwargs: Dict[str, Any] = dict(
            hidden_sizes=config["model_hidden_sizes"],
            encoder_feature_dim=config["encoder_feature_dim"],
            encoder_num_layers=config["encoder_num_layers"],
            encoder_num_filters=config["encoder_num_filters"],
            detach_encoder=config["detach_encoder"]
        )
        q_model = QofMuCnnModel
        q_model_kwargs = dict(
            hidden_sizes=config["model_hidden_sizes"],
            encoder_feature_dim=config["encoder_feature_dim"],
            encoder_num_layers=config["encoder_num_layers"],
            encoder_num_filters=config["encoder_num_filters"],
        )


    agent = SacAgent(
        ModelCls=pi_model,
        QModelCls=q_model,
        model_kwargs=pi_model_kwargs,
        q_model_kwargs=q_model_kwargs,
        # augmentations=augmentations,
    )
    
    run_slot, aff_code = remove_run_slot(
        encode_affinity(
            run_slot=0,
            n_cpu_core=config.get("n_cpu_core", 1),
            n_gpu=config.get("n_gpu", 1),
        )
    )
    print(run_slot, decode_affinity(aff_code))
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        seed=config["seed"],
        affinity=make_affinity(
            n_cpu_core=config.get("n_cpu_core", 1),
            n_gpu=config.get("n_gpu", 1),
            cpu_per_run=config.get("n_cpu_core", 1),
        ),
    )
    log_dir = f"/tmp/{config['name']}"
    with logger_context(log_dir, 0, config["name"], config, override_prefix=True):
        runner.train()


if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=True)
    setup_logger()
    experiment_id = config.pop("experiment_id")
    build_and_train(config)
