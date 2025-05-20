from sb3_contrib.ppo_recurrent.policies import ActorCriticLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gym
import torch

class CustomFeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        input_dim = observation_space.shape[0]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.fc(obs)

class CustomLSTMPolicy(ActorCriticLstmPolicy):
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,
            **kwargs
        )

    def forward(self,obs):
        return super().forward(obs)
