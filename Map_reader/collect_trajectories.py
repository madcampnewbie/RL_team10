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

def improved_collect_trajectory(env, policy, localization_module, guidance_module, 
                              device="cpu", max_steps=50, localization_trust=0.85):
    """
    Improved trajectory collection with better reward shaping and guidance handling
    """
    obs_list, act_list, loc_list, guide_list, act_onehot_list = [], [], [], [], []
    reward_list, done_list, logp_list, guidance_weight_list = [], [], [], []
    
    obs = env.reset()
    done = False
    h_actor = None
    h_local = None
    prev_action = torch.zeros(4, device=device)
    
    # Pre-compute true optimal policy
    true_guide = compute_policy_field(env.grid, env.goal)
    
    trajectory_reward = 0
    
    for step in range(max_steps):
        obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
        input_vec = torch.cat([prev_action, obs_flat], dim=-1).unsqueeze(0).unsqueeze(0)
        
        # Localization with uncertainty
        with torch.no_grad():
            logits, h_local = localization_module(input_vec, h_local)
            loc_probs = F.softmax(logits.squeeze(0), dim=-1)
            pred = torch.argmax(loc_probs, dim=-1)
            loc_confidence = loc_probs.max().item()
            
            pred_y = (pred // env.width).item()
            pred_x = (pred % env.width).item()
            loc_tensor = torch.tensor([[pred_y, pred_x]], dtype=torch.float32, device=device).unsqueeze(0) / 10.0
        
        # Get guidance with confidence weighting
        if loc_confidence < localization_trust:
            # Low confidence - provide weaker guidance
            guide_tensor = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32, device=device).view(1, 1, 4)
        else:
            guide_dir = guidance_module[pred_y][pred_x]
            if isinstance(guide_dir, int):
                guide_tensor = F.one_hot(torch.tensor(guide_dir, device=device), num_classes=4).float()
                # Add some noise to encourage exploration
                noise = torch.randn(4, device=device) * 0.1
                guide_tensor = (guide_tensor + noise).softmax(dim=-1).unsqueeze(0).unsqueeze(0)
            else:
                guide_tensor = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32, device=device).view(1, 1, 4)
        
        # Policy forward pass
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        prev_action_tensor = prev_action.unsqueeze(0).unsqueeze(0)
        
        logits, value, h_actor, guidance_weight = policy(
            obs_tensor, prev_action_tensor, loc_tensor, guide_tensor, h_actor
        )
        
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        logp = dist.log_prob(action)
        
        # Environment step
        obs_, reward, done, _ = env.step(action.item())
        
        # Improved reward shaping
        true_y, true_x = env.agent_pos  # Use actual position
        true_optimal_action = true_guide[true_y][true_x] if isinstance(true_guide[true_y][true_x], int) else -1
        
        # Multi-component reward
        goal_reward = 3.0 * reward  # Goal achievement
        
        # Progress reward (distance to goal)
        dist_before = np.sqrt((true_y - env.goal[0])**2 + (true_x - env.goal[1])**2)
        new_y, new_x = env.agent_pos
        dist_after = np.sqrt((new_y - env.goal[0])**2 + (new_x - env.goal[1])**2)
        progress_reward = 0.5 * (dist_before - dist_after)  # Positive if getting closer
        
        # Guidance following reward (scaled by localization confidence)
        if true_optimal_action >= 0:
            follow_reward = 0.3 * loc_confidence if action.item() == true_optimal_action else -0.1
        else:
            follow_reward = 0
        
        # Exploration penalty for hitting walls
        if (true_y, true_x) == (new_y, new_x) and action.item() != 4:  # Didn't move
            wall_penalty = -0.5
        else:
            wall_penalty = 0
        
        # Time penalty
        time_penalty = -0.3
        
        shaped_reward = goal_reward + progress_reward + follow_reward + wall_penalty + time_penalty
        trajectory_reward += shaped_reward
        
        # Store trajectory data
        obs_list.append(obs_tensor.squeeze())
        act_list.append(action.squeeze())
        act_onehot_list.append(F.one_hot(action, num_classes=4).float().squeeze())
        loc_list.append(loc_tensor.squeeze())
        guide_list.append(guide_tensor.squeeze())
        reward_list.append(torch.tensor(shaped_reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(done, dtype=torch.float32, device=device))
        logp_list.append(logp.detach().squeeze())
        guidance_weight_list.append(guidance_weight.squeeze())
        
        obs = obs_
        prev_action = F.one_hot(action, num_classes=4).float().squeeze(0)
        
        if done:
            # Success bonus
            trajectory_reward += 5.0
            reward_list[-1] += 5.0
            break
    
    # Convert lists to tensors with batch dimension
    return dict(
        obs=torch.stack(obs_list).unsqueeze(0),
        actions=torch.stack(act_list).unsqueeze(0),
        prev_actions=torch.stack(act_onehot_list).unsqueeze(0),
        loc=torch.stack(loc_list).unsqueeze(0),
        guide=torch.stack(guide_list).unsqueeze(0),
        rewards=torch.stack(reward_list).unsqueeze(0).unsqueeze(-1),
        dones=torch.stack(done_list).unsqueeze(0).unsqueeze(-1),
        logps=torch.stack(logp_list).unsqueeze(0),
        guidance_weights=torch.stack(guidance_weight_list).unsqueeze(0),
        total_reward=trajectory_reward,
        success=done
    )
