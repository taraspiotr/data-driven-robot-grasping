from mrunner.helpers.specification_helper import create_experiments_helper

config = {
    "name": "sample_traj",
    "num_traj": 100,
    "storage_dir": "/scratch/scratch/ucabpmt/traj/",
    "is_discrete": True,
}
name = globals()["script"][:-3]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="taraspiotr/data-driven-robot-grasping",
    script="python3.8 experiments/sample_random_trajectories.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid={"seed": list(range(1))},
)

