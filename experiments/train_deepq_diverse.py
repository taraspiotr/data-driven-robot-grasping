# from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

from grasp.sac.sac import sac
from grasp.sac.core import CNNActorCritic
from grasp.env import KukaDiverseObjectEnv

from baselines import deepq


def create_kuka_gym_diverse_env():
    return KukaDiverseObjectEnv(
        renders=False,
        isDiscrete=True,
        blockRandom=0,
        cameraRandom=0,
        numObjects=1,
        isTest=False,
        width=32,
        height=32,
        maxSteps=20,
    )


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl["episode_rewards"][-101:-1]) / 100
    totalt = lcl["t"]
    # print("totalt")
    # print(totalt)
    is_solved = totalt > 2000 and total >= 10
    return is_solved


if __name__ == "__main__":
    seed = 0

    # model = deepq.models.mlp([64])
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], dueling=False
    )
    act = deepq.learn(
        create_kuka_gym_diverse_env(),
        q_func=model,
        lr=1e-3,
        max_timesteps=10000000,
        buffer_size=50000,
        exploration_fraction=0.01,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
    )
    print("Saving model to kuka_model.pkl")
    act.save("kuka_model.pkl")
