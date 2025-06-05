import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import random

from env import generate_diverse_path, mutate_walls_nearby, GridEnv
from guidance_module import compute_policy_field, visualize_policy
from localization_module import train_localization_module
from collect_trajectories import improved_collect_trajectory


# ===== Improved Policy Network =====
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





# ===== GAE Computation =====
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation"""
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    for b in range(B):
        last_gae = 0
        for t in reversed(range(T)):
            next_value = values[b, t + 1] if t + 1 < T else 0
            mask = 1.0 - dones[b, t].item()
            delta = rewards[b, t] + gamma * next_value * mask - values[b, t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[b, t] = last_gae
            returns[b, t] = advantages[b, t] + values[b, t]
    
    return advantages.detach(), returns.detach()








# ===== Improved PPO Update =====
def improved_ppo_update(policy_net, optimizer, trajectories, 
                       clip_param=0.2, value_coef=0.5, entropy_coef=0.01,
                       max_grad_norm=0.5, ppo_epochs=4):
    """
    Improved PPO update with multiple epochs and gradient clipping
    """

    # Concatenate all trajectories
    obs = torch.cat([t["obs"] for t in trajectories], dim=1)
    prev_actions = torch.cat([t["prev_actions"] for t in trajectories], dim=1)
    loc = torch.cat([t["loc"] for t in trajectories], dim=1)
    guide = torch.cat([t["guide"] for t in trajectories], dim=1)
    actions = torch.cat([t["actions"] for t in trajectories], dim=1)
    rewards = torch.cat([t["rewards"] for t in trajectories], dim=1).squeeze(-1)
    dones = torch.cat([t["dones"] for t in trajectories], dim=1).squeeze(-1)
    old_logps = torch.cat([t["logps"] for t in trajectories], dim=1)
    
    # Compute advantages and returns once
    with torch.no_grad():
        _, old_values, _, _ = policy_net(obs, prev_actions, loc, guide)
        advantages, returns = compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Multiple epochs of PPO updates
    for _ in range(ppo_epochs):
        # Forward pass
        logits, values, _, guidance_weights = policy_net(obs, prev_actions, loc, guide)
        
        # Compute new log probs and entropy
        dist = torch.distributions.Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO loss
        ratio = (new_logps - old_logps).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Guidance regularization - encourage using guidance when confident
        guidance_reg = -guidance_weights.mean() * 0.1
        
        # Total loss
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + guidance_reg
        
        # Backward and clip gradients
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'guidance_weight': guidance_weights.mean().item()
    }






# ===== Curriculum Learning =====
class CurriculumScheduler:
    def __init__(self, start_mutation_rate=0.0, end_mutation_rate=0.2,                                                                                                                                                                                                                                                                                                  
                 warmup_episodes=2000, total_episodes=20000):
        self.start_rate = start_mutation_rate
        self.end_rate = end_mutation_rate
        self.warmup = warmup_episodes
        self.total = total_episodes
        
    def get_mutation_rate(self, episode):
        if episode < self.warmup:
            return self.start_rate
        
        progress = (episode - self.warmup) / (self.total - self.warmup)
        progress = min(1.0, progress)
        return self.start_rate + (self.end_rate - self.start_rate) * progress







# ===== Evaluation Functions =====
def evaluate_on_hard_envs(policy_net, localization_module, base_grid, goal, 
                         num_eval_episodes=20, device='cpu'):
    """Evaluate on environments with high mutation rates"""
    policy_net.eval()
    successes = 0
    
    for i in range(num_eval_episodes):
        # Create hard environment
        mutated_grid, reachable = mutate_walls_nearby(
            base_grid, goal, 
            mutation_rate=0.3,  # High mutation
            patch_size=5,       # Larger patches
            seed=1000 + i
        )
        guidance = compute_policy_field(mutated_grid, goal)
        env = GridEnv(mutated_grid, goal, reachable)
        
        # Run episode
        with torch.no_grad():
            trajectory = improved_collect_trajectory(
                env, policy_net, localization_module, guidance,
                device=device, max_steps=50
            )
        
        if trajectory['success']:
            successes += 1
    
    policy_net.train()
    return successes / num_eval_episodes










# ===== Main Training Function =====
def main():
    # Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Generate base map
    grid, goal, reachable_starts = generate_diverse_path(
        height=10, width=10, wall_prob=0.3, seed=42
    )
    guidance = compute_policy_field(grid, goal)
    
    print("Base map generated. Goal:", goal)
    visualize_policy(guidance)
    
    # Train localization module
    print("\nTraining localization module...")
    localization_module = train_localization_module(grid, goal, reachable_starts, visualize=False)
    localization_module = localization_module.to(DEVICE)
    localization_module.eval()
    for param in localization_module.parameters():
        param.requires_grad = False
    
    # Initialize policy network
    policy_net = ImprovedPolicyNet().to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    
    # Training configuration
    num_episodes = 10000
    batch_size = 16  # Collect more trajectories before update
    update_frequency = batch_size
    
    # Curriculum learning
    curriculum = CurriculumScheduler(
        start_mutation_rate=0.1,
        end_mutation_rate=0.3,
        warmup_episodes=1000,
        total_episodes=num_episodes
    )
    
    # Tracking
    episode_rewards = []
    success_rate_window = deque(maxlen=100)
    best_success_rate = 0
    
    # Storage for trajectories
    trajectory_buffer = []
    
    print("\nStarting training...")
    
    for episode in range(num_episodes):
        # Curriculum-based environment mutation
        mutation_rate = curriculum.get_mutation_rate(episode)
        
        # Generate new environment variant every few episodes
        if episode % 5 == 0:
            mutated_grid, mutated_reachable = mutate_walls_nearby(
                grid, goal, 
                mutation_rate=mutation_rate, 
                patch_size=3,
                seed=episode  # Different seed for variety
            )
            guidance = compute_policy_field(mutated_grid, goal)
            env = GridEnv(mutated_grid, goal, mutated_reachable)
        
        # Collect trajectory
        trajectory = improved_collect_trajectory(
            env, policy_net, localization_module, guidance, 
            device=DEVICE, max_steps=50
        )
        
        #print(trajectory["obs"].shape)

        trajectory_buffer.append(trajectory)
        episode_rewards.append(trajectory['total_reward'])
        success_rate_window.append(1.0 if trajectory['success'] else 0.0)
        
        # Update policy
        if len(trajectory_buffer) >= update_frequency:
            # Perform PPO update
            metrics = improved_ppo_update(
                policy_net, optimizer, trajectory_buffer,
                clip_param=0.2, value_coef=0.5, entropy_coef=0.01,
                max_grad_norm=0.5, ppo_epochs=4
            )
            
            # Clear buffer
            trajectory_buffer = []
            
            # Logging
            if (episode+1) % (update_frequency*10) == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                success_rate = np.mean(list(success_rate_window))
                
                print(f"\n[Episode {episode}/{num_episodes}]")
                print(f"  Mutation Rate: {mutation_rate:.3f}")
                print(f"  Avg Reward (100 ep): {avg_reward:.2f}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Guidance Weight: {metrics['guidance_weight']:.3f}")
                
                # Save best model
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    torch.save({
                        'episode': episode,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'success_rate': success_rate,
                        'avg_reward': avg_reward
                    }, 'best_navigation_model.pth')
                    print(f"  💾 New best model saved! (Success rate: {success_rate:.2%})")
        
        # Periodic evaluation on harder environments
        if episode % 500 == 0 and episode > 0:
            print("\n🧪 Evaluating on hard environments...")
            eval_success_rate = evaluate_on_hard_envs(
                policy_net, localization_module, grid, goal, 
                num_eval_episodes=20, device=DEVICE
            )
            print(f"  Hard environment success rate: {eval_success_rate:.2%}")
    
    print("\n✅ Training completed!")
    print(f"Best success rate achieved: {best_success_rate:.2%}")


if __name__ == "__main__":
    main()