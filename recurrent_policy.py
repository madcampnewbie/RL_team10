from sb3_contrib.ppo_recurrent.policies import ActorCriticLstmPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gym
import torch

class CustomFeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space,features_dim=1):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]*observation_space.shape[1]
        
        self.conv = nn.Conv2d(input_dim, 32, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(32*2*2, features_dim) #Assuming 3 by 3 conv input

        self.act = nn.ReLU()

        self._features_dim = features_dim

    def forward(self, obs):
        out = self.conv(obs)
        out = self.act(self.bn1(out))
        out = self.fc1(self.flt(out))
        return out

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
