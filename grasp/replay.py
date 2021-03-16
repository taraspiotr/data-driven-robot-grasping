from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer


class UniformRadReplayBuffer(UniformReplayBuffer):
    def __init__(self, *args, augs_func, **kwargs):
        super().__init__(*args, **kwargs)
        self.augs_func = augs_func

    def sample_batch(self, batch_B):
        batch = super().sample_batch(batch_B)
        observation = batch.agent_inputs.observation
        for aug, func in self.augs_func.items():
            observation = func(observation)
        agent_inputs = batch.agent_inputs._replace(observation=observation)
        batch = batch._replace(agent_inputs=agent_inputs)

        observation = batch.target_inputs.observation
        for aug, func in self.augs_func.items():
            observation = func(observation)
        target_inputs = batch.target_inputs._replace(observation=observation)
        batch = batch._replace(target_inputs=target_inputs)
        return batch
