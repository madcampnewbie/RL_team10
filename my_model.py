import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
import numpy as np

from env import GridEnv

class MapFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=32):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
policy_kwargs = dict(
    features_extractor_class=MapFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=32),
    
    net_arch=dict(
        pi=[64, 128, 256], 
        vf=[64, 128, 256]
    ),
    
    lstm_hidden_size=256,
    n_lstm_layers=2,
    shared_lstm=False,
    enable_actor_lstm=True,
    enable_critic_lstm=True,
    lstm_dropout=0.2,
    
    lstm_kwargs=dict(
        proj_size=None,
        bidirectional=False,
        bias=True
    )
)


env = DummyVecEnv([lambda: GridEnv()]) 

model = RecurrentPPO(
    policy=RecurrentActorCriticPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    gamma=0.99,
    learning_rate=1e-3,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.5,
    clip_range=0.2,
)
model.learn(total_timesteps=10_000, progress_bar=True)

obs = env.reset()
lstm_states = None
episode_starts = np.ones((env.num_envs,), dtype=bool)
# Save the model
model.save("ppo_recurrent")

