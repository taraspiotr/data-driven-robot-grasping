from mrunner.helpers.specification_helper import create_experiments_helper

config = {
    "name": "sac_kuka_diverse",
    "env_num_objects": 5,
    "env_camera_random": 0,
    "env_use_height_hack": True,
    "model_hidden_sizes": (256, 256),
    "encoder_num_filters": 32,
    "cuda_idx": 0,
    "learning_rate": 3e-3,
    "alpha": None,
    "env_block_random": 0,
    "encoder_num_layers": 2,
    "encoder_feature_dim": 32,
    "augmentations": [],
    "observation_size": 64,
    "batch_B": 8,
    "eval_n_envs": 1,
    "n_cpu_core": 8,
    "n_gpu": 1,
}

params_grid = {}
name = globals()["script"][:-3]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="taraspiotr/data-driven-robot-grasping",
    script="python3.8 experiments/sac.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
)

