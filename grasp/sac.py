import torch

from rlpyt.algos.qpg.sac import SAC as _SAC
from rlpyt.algos.qpg.sac import OptInfo

class SAC(_SAC):
    pass
    # def optim_initialize(self, rank):
    #     super().optim_initialize(rank=rank)
    #     self.encoder_optimizer = self.OptimCls(self.agent.encoder_parameters(),
    #         lr=self.learning_rate, **self.optim_kwargs)

    # def optimize_agent(self, itr, samples=None, sampler_itr=None):
    #     """
    #     Extracts the needed fields from input samples and stores them in the 
    #     replay buffer.  Then samples from the replay buffer to train the agent
    #     by gradient updates (with the number of updates determined by replay
    #     ratio, sampler batch size, and training batch size).
    #     """
    #     itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
    #     if samples is not None:
    #         samples_to_buffer = self.samples_to_buffer(samples)
    #         self.replay_buffer.append_samples(samples_to_buffer)
    #     opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
    #     if itr < self.min_itr_learn:
    #         return opt_info
    #     for _ in range(self.updates_per_optimize):
    #         samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
    #         losses, values = self.loss(samples_from_replay)
    #         q1_loss, q2_loss, pi_loss, alpha_loss = losses

    #         if alpha_loss is not None:
    #             self.alpha_optimizer.zero_grad()
    #             alpha_loss.backward()
    #             self.alpha_optimizer.step()
    #             self._alpha = torch.exp(self._log_alpha.detach())

    #         self.encoder_optimizer.zero_grad()
    #         self.pi_optimizer.zero_grad()
    #         self.q1_optimizer.zero_grad()
    #         self.q2_optimizer.zero_grad()

    #         loss = pi_loss + q1_loss + q2_loss
    #         loss.backward()

    #         # pi_loss.backward(retain_graph=True)
    #         pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(),
    #             self.clip_grad_norm)

    #         # Step Q's last because pi_loss.backward() uses them?
    #         # q1_loss.backward(retain_graph=True)
    #         q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(),
    #             self.clip_grad_norm)

    #         # q2_loss.backward()
    #         q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(),
    #             self.clip_grad_norm)

    #         self.pi_optimizer.step()
    #         self.q1_optimizer.step()
    #         self.q2_optimizer.step()
    #         self.encoder_optimizer.step()

    #         grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm)

    #         self.append_opt_info_(opt_info, losses, grad_norms, values)
    #         self.update_counter += 1
    #         if self.update_counter % self.target_update_interval == 0:
    #             self.agent.update_target(self.target_update_tau)

    #     return opt_info