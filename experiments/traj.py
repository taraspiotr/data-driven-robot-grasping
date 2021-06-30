import os
import pickle

import numpy as np
from mrunner.helpers.client_helper import get_configuration


def main(traj_dir, **kwargs):
    trajs = os.listdir(traj_dir)[:2]
    for traj in trajs:
        print(traj)
        with open(os.path.join(traj_dir, traj), "rb") as f:
            t = pickle.load(f)
    print("Loaded all")


if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=False)
    main(**config)
