import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import generate_diverse_path, mutate_walls_nearby,GridEnv
from guidance_module import compute_policy_field, visualize_policy
from localization_module import train_localization_module
from collect_trajectories import collect_trajectory, get_dijkstra_direction

class PolicyNetV2(nn.Module):
    def __init__(self, obs_dim=3*3, action_dim=4, hidden_dim=256):  # 128 -> 256
        super().__init__()
        input_dim = obs_dim + action_dim + 2 + 4
        
        # 2층 GRU로 확장
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # 더 깊은 헤드
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, obs_seq, act_seq, loc_seq, guide_seq, hidden_state=None):
        B, T, H, W = obs_seq.shape
        obs_flat = obs_seq.view(B, T, -1)
        act_seq = act_seq.view(B, T, -1)
        x = torch.cat([obs_flat, act_seq, loc_seq, guide_seq], dim=-1)
        
        out, hidden = self.gru(x, hidden_state)
        logits = self.actor_head(out)
        values = self.critic_head(out).squeeze(-1)
        return logits, values, hidden

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=3*3, action_dim=4, hidden_dim=128):
        super().__init__()
        input_dim = obs_dim + action_dim + 2 + 4  # 
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, act_seq, loc_seq, guide_seq, hidden_state=None):   # obs_seq는 (B,T,3,3) 형태, action_seq : (B,T,4), loc_seq는 (B,T,2)로 위치추정, guide_seq는 (B,T,4)
        
        B, T, H, W = obs_seq.shape
        obs_flat = obs_seq.view(B, T, -1)
        act_seq = act_seq.view(B,T, -1)                                           # 이걸 flatten
        x = torch.cat([obs_flat, act_seq, loc_seq, guide_seq], dim=-1)              # 다시 condat
        out, hidden = self.gru(x, hidden_state)
        logits = self.actor_head(out)
        values = self.critic_head(out).squeeze(-1)
        return logits, values, hidden


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Args:
        rewards: (B, T)
        values: (B, T)
        dones: (B, T)
    Returns:
        advantages, returns: (B, T)
    """
    B, T = rewards.shape
    advantages = torch.zeros((B, T), dtype=torch.float32).to(rewards.device)
    returns = torch.zeros((B, T), dtype=torch.float32).to(rewards.device)
    
    for b in range(B):
        last_gae = 0
        for t in reversed(range(T)):
            next_value = values[b, t + 1] if t + 1 < T else 0
            mask = 1.0 - dones[b, t].item()
            delta = rewards[b, t].item() + gamma * next_value * mask - values[b, t].item()
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[b, t] = last_gae
            returns[b, t] = advantages[b, t] + values[b, t]
    
    return advantages.detach(), returns.detach()

def test_optimal_path(env, guidence):
    print(f"Start: {env.agent_pos}, Goal: {env.goal}")
    
    for step in range(50):
        optimal_action = guidence[env.agent_pos[0]][env.agent_pos[1]]
        #optimal_action = get_dijkstra_direction(env.grid, env.goal, env.agent_pos)
        print(f"Step {step}: Pos {env.agent_pos}, Action {optimal_action}")
        
        obs, reward, done, _ = env.step(optimal_action)
        
        if done:
            print(f"Reached goal in {step+1} steps! Reward: {reward}")
            return True
            
    print("Failed to reach goal with optimal policy!")
    return False




def ppo_update(policy_net, optimizer, batch, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    batch: dict containing:
        - obs: (B, T, 3, 3)
        - prev_actions: (B, T, 4)
        - loc: (B, T, 2)
        - guide: (B, T, 4)
        - actions: (B, T)
        - rewards: (B, T, 1)
        - dones: (B, T, 1)
        - logps: (B, T)
    """
    obs = batch["obs"]
    prev_actions = batch["prev_actions"]
    loc = batch["loc"]
    guide = batch["guide"]
    actions = batch["actions"]
    rewards = batch["rewards"].squeeze(-1)  # (B, T, 1) -> (B, T)
    dones = batch["dones"].squeeze(-1)      # (B, T, 1) -> (B, T)
    old_logps = batch["logps"]
    
    #print(f"obs: {obs.shape}, prev_actions: {prev_actions.shape}, loc: {loc.shape}, guide: {guide.shape}")
    #print(f"actions: {actions.shape}, rewards: {rewards.shape}, dones: {dones.shape}, old_logps: {old_logps.shape}")
    
    B, T, _, _ = obs.shape

    # 정책 네트워크 forward
    logits, values, _ = policy_net(obs, prev_actions, loc, guide)  # logits: (B, T, 4), values: (B, T)

    # ppo_update에서 loss 계산 전에 추가
    #print(f"Action distribution: {torch.bincount(actions.flatten(), minlength=4)}")
    #print(f"Logits range: {logits.min():.3f} ~ {logits.max():.3f}")
    #print(f"Values range: {values.min():.3f} ~ {values.max():.3f}")
    
    # 확률 분포 및 로그 확률 계산
    dist = torch.distributions.Categorical(logits=logits)
    new_logps = dist.log_prob(actions)  # (B, T)
    entropy = dist.entropy().mean()

    # GAE 계산
    advantages, returns = compute_gae(rewards, values, dones)  # (B, T), (B, T)

    # PPO 손실 계산
    ratio = (new_logps - old_logps).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.item()

def analyze_episode_quality(data):
    total_reward = data["rewards"].sum().item()
    episode_length = data["rewards"].shape[1]
    
    print(f"Episode reward: {total_reward:.3f}, Length: {episode_length}")
    
    # 액션별 보상 분석
    actions = data["actions"].squeeze(0)  # (T,)
    rewards = data["rewards"].squeeze(0).squeeze(-1)  # (T,)
    
    for action in range(4):
        action_mask = (actions == action)
        if action_mask.sum() > 0:
            avg_reward = rewards[action_mask].mean()
            print(f"Action {action}: count={action_mask.sum()}, avg_reward={avg_reward:.3f}")


def compute_losses(policy_net, batch, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    PPO 손실들을 계산하되 backward는 하지 않음
    """
    obs = batch["obs"]
    prev_actions = batch["prev_actions"]
    loc = batch["loc"]
    guide = batch["guide"]
    actions = batch["actions"]
    rewards = batch["rewards"].squeeze(-1)  # (B, T, 1) -> (B, T)
    dones = batch["dones"].squeeze(-1)      # (B, T, 1) -> (B, T)
    old_logps = batch["logps"]
    
    B, T, _, _ = obs.shape

    # 정책 네트워크 forward
    logits, values, _ = policy_net(obs, prev_actions, loc, guide)
    
    # 확률 분포 및 로그 확률 계산
    dist = torch.distributions.Categorical(logits=logits)
    new_logps = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # GAE 계산
    advantages, returns = compute_gae(rewards, values, dones)

    # PPO 손실 계산
    ratio = (new_logps - old_logps).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)

    return policy_loss, value_loss, entropy


def main():
    # 지도가 주어짐
    grid, goal, reachable_starts = generate_diverse_path(height=10, width=10, wall_prob=0.3, seed = 42)
    guidence = compute_policy_field(grid, goal)

    visualize_policy(guidence)

    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    
    localization_module = train_localization_module(grid, goal, reachable_starts, visualize=False)
    localization_module.eval()
    for param in localization_module.parameters():
        param.requires_grad = False

    policy_net = PolicyNetV2().to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    
    # 배치 설정
    batch_size = 10
    collected_episodes = []
    
    # 이동 평균 추적
    reward_history = []
    moving_avg_window = 100


    num_episodes = 10000
    for episode in range(num_episodes):
        if episode % 10 ==0:
            mutated_grid, reachable_starts = mutate_walls_nearby(grid, goal, mutation_rate=0.2, patch_size=3)
            env = GridEnv(mutated_grid, goal, reachable_starts)
    
        # trajectory 수집
        data = collect_trajectory(env, policy_net, localization_module, guidence, DEVICE)
        collected_episodes.append(data)
        #analyze_episode_quality(data)
        # PPO 업데이트
        policy_loss, value_loss, entropy = ppo_update(policy_net, optimizer, data)

        # 보상 기록
        episode_reward = data["rewards"].sum().item()
        reward_history.append(episode_reward)

        if len(collected_episodes) >= batch_size:
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            # 각 에피소드별로 손실 계산 및 그래디언트 누적
            for ep_data in collected_episodes:
                policy_loss, value_loss, entropy = compute_losses(policy_net, ep_data)
                
                # 전체 손실 계산
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                # 그래디언트 누적 (배치 크기로 나누어서 평균화)
                (loss / batch_size).backward()
                
                # 통계 누적
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
            
            # 누적된 그래디언트로 한 번에 업데이트
            optimizer.step()
            
            # 평균 손실 계산
            avg_policy_loss = total_policy_loss / batch_size
            avg_value_loss = total_value_loss / batch_size
            avg_entropy = total_entropy / batch_size
            
            # 이동 평균 계산
            if len(reward_history) >= moving_avg_window:
                moving_avg = sum(reward_history[-moving_avg_window:]) / moving_avg_window
            else:
                moving_avg = sum(reward_history) / len(reward_history)
            
            # 로깅
            recent_rewards = [ep["rewards"].sum().item() for ep in collected_episodes]
            batch_avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            if (episode + 1) % (batch_size * 5) == 0:  # 50 에피소드마다 출력
                print(f"[{episode+1}/{num_episodes}] "
                      f"Batch Avg Reward: {batch_avg_reward:.2f} | "
                      f"Moving Avg ({moving_avg_window}): {moving_avg:.2f} | "
                      f"π_loss: {avg_policy_loss:.3f} | "
                      f"V_loss: {avg_value_loss:.3f} | "
                      f"Entropy: {avg_entropy:.3f}")
            
            # 배치 초기화
            collected_episodes = []
        



if __name__ == "__main__":
    num_episodes = 10000
    value_coef = 0.5
    entropy_coef = 0.01
    main()