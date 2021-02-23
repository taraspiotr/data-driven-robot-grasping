from typing import Dict, Any

import neptune
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from grasp.env import create_kuka_gym_diverse_env
from grasp.logger import setup_logger
from grasp.policies import PiCnnModel
from grasp.value_functions import QofMuCnnModel
from grasp.collectors import SerialEvalCollectorLogger


def build_and_train(config: Dict[str, Any]):

    env_kwargs: Dict[str, Any] = dict(
        num_objects=config["env_num_objects"],
        camera_random=config["env_camera_random"],
        block_random=config["env_block_random"],
        use_height_hack=config["env_use_height_hack"],
    )
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
    )  # Run with defaults.

    model_kwargs: Dict[str, Any] = dict(
        hidden_sizes=config["model_hidden_sizes"],
        encoder_feature_dim=config["encoder_feature_dim"],
        encoder_num_layers=config["encoder_num_layers"],
        encoder_num_filters=config["encoder_num_filters"],
    )
    agent = SacAgent(
        ModelCls=PiCnnModel,
        QModelCls=QofMuCnnModel,
        model_kwargs=model_kwargs,
        q_model_kwargs=model_kwargs,
    )
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=config["cuda_idx"]),
        seed=config["seed"],
    )
    log_dir = f"/tmp/{config['name']}"
    with logger_context(log_dir, 0, config["name"], config, override_prefix=True):
        runner.train()


if __name__ == "__main__":

    config = {
        "name": "sac_kuka_diverse",
        "alpha": 1e-3,
        "learning_rate": 3e-4,
        "env_num_objects": 5,
        "env_camera_random": 0,
        "env_block_random": 0,
        "env_use_height_hack": True,
        "model_hidden_sizes": (256, 256),
        "encoder_feature_dim": 32,
        "encoder_num_layers": 2,
        "encoder_num_filters": 32,
        "cuda_idx": 0,
        "seed": 0,
    }

    neptune.init(project_qualified_name="taraspiotr/data-driven-robot-grasping")
    neptune.create_experiment(name=config["name"], params=config)
    setup_logger()
    build_and_train(config)