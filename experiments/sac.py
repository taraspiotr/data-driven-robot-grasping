from typing import Optional, Dict

import neptune
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from grasp.env import create_kuka_gym_diverse_env
from grasp.logger import setup_logger


def build_and_train(
    env_id: str = "Hopper-v3", run_ID: int = 0, cuda_idx: Optional[int] = None
):
    sampler = SerialSampler(
        EnvCls=create_kuka_gym_diverse_env,
        env_kwargs=dict(test=False),
        eval_env_kwargs=dict(test=True),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )
    algo = SAC(
        bootstrap_timelimit=False, min_steps_learn=int(1e3)
    )  # Run with defaults.
    agent = SacAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config: Dict[str, str] = dict(env_id=env_id)
    name = "sac_" + env_id
    log_dir = "/tmp/example_2"
    with logger_context(log_dir, run_ID, name, config, override_prefix=True):
        runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env_id", help="environment ID", default="Hopper-v3")
    parser.add_argument(
        "--run_ID", help="run identifier (logging)", type=int, default=0
    )
    parser.add_argument("--cuda_idx", help="gpu to use ", type=int, default=None)
    args = parser.parse_args()

    neptune.init(project_qualified_name="taraspiotr/data-driven-robot-grasping")
    neptune.create_experiment(name="rlpyt_test", params=vars(args))
    setup_logger()
    build_and_train(
        env_id=args.env_id, run_ID=args.run_ID, cuda_idx=args.cuda_idx,
    )
