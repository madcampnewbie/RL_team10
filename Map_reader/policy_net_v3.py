import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedPolicyNet(nn.Module):
    def __init__(self, obs_dim=3*3, action_dim=4, hidden_dim=256):
        super().__init__()
        
        # Separate processing streams
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.guidance_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.loc_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Attention mechanism for guidance
        self.guidance_attention = nn.Linear(hidden_dim, 1)
        
        # Combined processing
        combined_dim = 64 + 32 + 32 + 4  # obs + guidance + loc + prev_action
        self.gru = nn.GRU(combined_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # Separate heads with guidance influence
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim//2),  # +4 for guidance
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, obs_seq, act_seq, loc_seq, guide_seq, hidden_state=None):
        B, T, H, W = obs_seq.shape
        
        # Encode different inputs
        obs_flat = obs_seq.view(B, T, -1)
        obs_encoded = self.obs_encoder(obs_flat)
        guide_encoded = self.guidance_encoder(guide_seq)
        loc_encoded = self.loc_encoder(loc_seq)
        
        # Combine all features
        x = torch.cat([obs_encoded, guide_encoded, loc_encoded, act_seq], dim=-1)
        
        # GRU processing
        gru_out, hidden = self.gru(x, hidden_state)
        
        # Compute guidance attention weight
        guidance_weight = torch.sigmoid(self.guidance_attention(gru_out))
        
        # Actor with guidance influence
        actor_input = torch.cat([gru_out, guide_seq * guidance_weight], dim=-1)
        logits = self.actor_head(actor_input)
        
        # Add guidance bias to logits (encourage following guidance)
        guidance_bias = guide_seq * 2.0  # Amplify guidance signal
        logits = logits + guidance_bias
        
        # Critic (no direct guidance influence)
        values = self.critic_head(gru_out).squeeze(-1)
        
        return logits, values, hidden, guidance_weight