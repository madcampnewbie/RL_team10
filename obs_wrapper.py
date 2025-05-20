import gym
import numpy as np
import torch

class AugmentedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        extra_info = np.random.randn(self.extra_dim)  
        return torch.cat((obs, extra_info), dim=1)
