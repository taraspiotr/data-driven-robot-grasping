from mrunner.helpers.specification_helper import create_experiments_helper

config = {"traj_dir": "/scratch/scratch/ucabpmt/traj/"}

params_grid = {}
name = globals()["script"][:-3]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="taraspiotr/data-driven-robot-grasping",
    script="python3.8 experiments/traj.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
)

