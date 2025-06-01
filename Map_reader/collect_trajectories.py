import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from scipy.ndimage import distance_transform_edt
from torch.distributions import Categorical
from guidance_module import compute_policy_field

def get_dijkstra_direction(grid, goal, pos):
    h, w = grid.shape
    cost = (grid != 1).astype(np.uint8)  # 통로는 1, 벽은 0
    distance = distance_transform_edt(cost == 1)
    gy, gx = goal
    goal_mask = np.zeros_like(cost)
    goal_mask[gy, gx] = 1
    distmap = distance_transform_edt(cost == 1, return_distances=True, return_indices=False)
    
    y, x = pos
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    best_dir = None
    min_d = float('inf')
    for idx, (dy, dx) in enumerate(directions):
        ny, nx = y+dy, x+dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] != 1:
            d = np.linalg.norm([ny - gy, nx - gx])
            if d < min_d:
                best_dir = idx
                min_d = d
    return best_dir  # 0~3

def collect_trajectory(env, policy, localization_module, guidance_module, device="cpu", max_steps=50):
    obs_list, act_list, loc_list, guide_list, act_onehot_list, reward_list, done_list, logp_list = [], [], [], [], [], [], [], []

    obs = env.reset()
    done = False
    h_actor = None
    h_local = None
    prev_action = torch.zeros(4, device=device)  # device 명시
    total_reward = 0
    true_guide = compute_policy_field(env.grid, env.goal)
    for _ in range(max_steps):
        obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)  # device 명시
        
        input_vec = torch.cat([prev_action.squeeze(0), obs_flat], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,13)

        # localization
        with torch.no_grad():
            logits, h_local = localization_module(input_vec, h_local)
            pred = torch.argmax(logits.squeeze(0), dim=-1)  # (1,)
            pred_y = (pred // env.width).item()
            pred_x = (pred % env.width).item()
            loc_tensor = torch.tensor([[pred_y, pred_x]], dtype=torch.float32, device=device).unsqueeze(0) / 10.0  # device 명시

        # guidance
        guide_dir = guidance_module[pred_y][pred_x]
        if isinstance(guide_dir, int):
            guide_tensor = F.one_hot(torch.tensor(guide_dir, device=device), num_classes=4).float().unsqueeze(0).unsqueeze(0)  # device 명시
        elif isinstance(guide_dir, str):
            guide_tensor = torch.tensor([0, 0, 0, 0], dtype=torch.float32, device=device).view(1, 1, 4)  # device 명시
        


        # action selection
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # device 명시
        prev_action_tensor = prev_action.unsqueeze(0).unsqueeze(0)  # 이미 device에 있음

        logits, value, h_actor = policy(obs_tensor, prev_action_tensor, loc_tensor, guide_tensor, h_actor)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        logp = dist.log_prob(action)
        
        # step
        obs_, reward, done, _ = env.step(action.item())

        guide_dir = guidance_module[pred_y][pred_x]
        true_dir = true_guide[pred_y][pred_x]
        #print(f"Guide dir: {guide_dir}, True optimal dir: {true_dir}, Agent action: {action.item()}")

        # reward shaping
        true_dir = true_guide[pred_y][pred_x]
        follow_reward = 1.0 if action.item() == true_dir else 0.0
        shaped_reward = 10.0*reward + follow_reward - 0.1  # step penalty
        total_reward += shaped_reward

        # 저장 - 모든 텐서를 같은 device에 생성
        obs_list.append(obs_tensor.squeeze(0).squeeze(0))  # (3, 3) - 이미 device에 있음
        act_list.append(action.squeeze())  # 이미 device에 있음
        act_onehot_list.append(F.one_hot(action, num_classes=4).float().squeeze())  # 이미 device에 있음
        loc_list.append(loc_tensor.squeeze(0).squeeze(0))  # (2,) - 이미 device에 있음
        guide_list.append(guide_tensor.squeeze(0).squeeze(0))  # (4,) - 이미 device에 있음
        reward_list.append(torch.tensor(shaped_reward, dtype=torch.float32, device=device))  # device 명시!
        done_list.append(torch.tensor(done, dtype=torch.float32, device=device))  # device 명시!
        logp_list.append(logp.detach().squeeze())  # 이미 device에 있음

        obs = obs_
        prev_action = F.one_hot(action, num_classes=4).float()  # 이미 device에 있음
        if done:
            break
    
    # collect_trajectory 마지막에 이거 추가해보세요
    #print(f"Episode length: {len(reward_list)}")
    #print(f"Total reward: {total_reward}")
    #print(f"Rewards: {[r.item() for r in reward_list[:10]]}")  # 처음 10개만
    #print(f"Final env state - Goal: {env.goal}, Agent: {env.agent_pos}")

    # 배치 차원을 추가하여 (1, T, ...) 형태로 만들기
    return dict(
        obs=torch.stack(obs_list).unsqueeze(0),           # (1, T, 3, 3)
        actions=torch.stack(act_list).unsqueeze(0),       # (1, T)
        prev_actions=torch.stack(act_onehot_list).unsqueeze(0),  # (1, T, 4)
        loc=torch.stack(loc_list).unsqueeze(0),           # (1, T, 2)
        guide=torch.stack(guide_list).unsqueeze(0),       # (1, T, 4)
        rewards=torch.stack(reward_list).unsqueeze(0).unsqueeze(-1),    # (1, T, 1)
        dones=torch.stack(done_list).unsqueeze(0).unsqueeze(-1),        # (1, T, 1)
        logps=torch.stack(logp_list).unsqueeze(0),        # (1, T)
        total_reward=total_reward
    )

