from mrunner.helpers.specification_helper import create_experiments_helper

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
}

params_grid = {
    "seed": [0, 1],
}
name = globals()["script"][:-3]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="taraspiotr/data-driven-robot-grasping",
    script="python3.7 experiments/sac.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
)

