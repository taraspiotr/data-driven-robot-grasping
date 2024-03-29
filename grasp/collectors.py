import imageio
import numpy as np

from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from mrunner.helpers import client_helper
from neptune.new.types import File


def log_trajectories(trajectories: np.ndarray) -> None:
    fn = "/tmp/trajectories.gif"
    imageio.mimsave(fn, trajectories.transpose(0, 2, 3, 1), duration=0.5)
    client_helper.experiment_["trajectories"].log(File(fn))


class SerialEvalCollectorLogger(SerialEvalCollector):
    def collect_evaluation(self, itr):

        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(
            self.envs[0].action_space.null_value(), len(self.envs)
        )
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)

        observed_traj = []
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                print(action[b])
                o, r, d, env_info = env.step(action[b])
                if b == 0:
                    observed_traj.append(o)
                traj_infos[b].step(
                    observation[b], action[b], r, d, agent_info[b], env_info
                )
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (
                self.max_trajectories is not None
                and len(completed_traj_infos) >= self.max_trajectories
            ):
                logger.log(
                    "Evaluation reached max num trajectories "
                    f"({self.max_trajectories})."
                )
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps " f"({self.max_T}).")
        # log_trajectories(np.stack(observed_traj))
        return completed_traj_infos
