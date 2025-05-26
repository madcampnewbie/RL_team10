import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class MemoryExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=1)
        H, W = observation_space["original"].shape

        # (1) local 3×3
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 2), nn.ReLU(), nn.Flatten()
        )

        # (2) original 15×15
        self.orig_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )

        # (3) memory + act_mem (4개 채널) → 총 5채널 CNN
        self.mem_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            local_dim = self.local_cnn(torch.zeros(1, 1, 3, 3)).shape[1]
            orig_dim  = self.orig_cnn(torch.zeros(1, 1, H, W)).shape[1]
            mem_dim   = self.mem_cnn(torch.zeros(1, 5, H, W)).shape[1]
        self._features_dim = local_dim + orig_dim + mem_dim

    def forward(self, obs):
        local    = obs["local"].unsqueeze(1).float()           # (B,1,3,3)
        original = obs["original"].unsqueeze(1).float()        # (B,1,H,W)

        mem_map  = obs["memory"].float().unsqueeze(1)          # (B,1,H,W)
        act_map  = obs["act_mem"].float()                      # (B,H,W,4)
        act_map[act_map == -1] = 0                             # 마스킹 처리 후
        act_map  = act_map.permute(0, 3, 1, 2) / 200.0         # (B,4,H,W), normalize to [0,1]

        mem_stack = torch.cat([mem_map, act_map], dim=1)       # (B,5,H,W)

        local_f = self.local_cnn(local)
        orig_f  = self.orig_cnn(original)
        mem_f   = self.mem_cnn(mem_stack)

        return torch.cat([local_f, orig_f, mem_f], dim=1)

