import numpy as np
import random
import gymnasium as gym                  
from gymnasium import spaces             
from gymnasium.spaces import Dict        
from collections import deque

# ------------------------------------------------------------
# 맵 생성기: 경로가 존재하는 원본 맵 생성
# ------------------------------------------------------------

def is_path_exists(grid, start, goal):
    from collections import deque
    h, w = grid.shape
    visited = set()
    queue = deque([start])

    while queue:
        y, x = queue.popleft()
        if (y, x) == goal:
            return True
        if (y, x) in visited:
            continue
        visited.add((y, x))
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] != 1:
                queue.append((ny, nx))
    return False


def generate_diverse_path(height, width, wall_prob):
    while True:
        grid = np.zeros((height, width), dtype=np.int32)

        # 1. 무작위 벽 배치
        for y in range(height):
            for x in range(width):
                if random.random() < wall_prob:
                    grid[y, x] = 1  # 벽

        # 2. 목표 위치 설정 (벽이 아닌 곳)
        while True:
            goal = (random.randint(0, height - 1), random.randint(0, width - 1))
            if grid[goal] == 0:
                grid[goal] = 2
                break

        # 3. goal까지 도달 가능한 지점들 계산
        reachable = []
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 0 and is_path_exists(grid, (y, x), goal):
                    reachable.append((y, x))

        # 4. 유효한 시작점이 충분하다면 맵 채택
        if len(reachable) > 0:
            return grid, goal, reachable


# ------------------------------------------------------------
# 변형기: 일부 벽을 a x a 이내에서 이동시킴
# ------------------------------------------------------------
def mutate_walls_nearby(grid, mutation_rate=0.2, patch_size=3):
    mutated = grid.copy()
    wall_positions = list(zip(*np.where(mutated == 1)))
    num_mutate = int(len(wall_positions) * mutation_rate)
    selected = random.sample(wall_positions, num_mutate)

    for y, x in selected:
        mutated[y, x] = 0  # 기존 벽 제거
        dy = random.randint(-(patch_size // 2), patch_size // 2)
        dx = random.randint(-(patch_size // 2), patch_size // 2)
        ny, nx = y + dy, x + dx
        if (0 <= ny < grid.shape[0]) and (0 <= nx < grid.shape[1]) and mutated[ny, nx] == 0:
            mutated[ny, nx] = 1  # 새 위치에 벽 추가

    return mutated

# ------------------------------------------------------------
# 강화학습 환경 클래스
# ------------------------------------------------------------
class GridEnv(gym.Env):
    def __init__(self, grid, goal, reachable_starts, original, render_mode=None):
        super(GridEnv, self).__init__()
        self.grid = grid
        self.goal = goal
        self.original = original
        self.render_mode = render_mode

        self.memory_map = np.full_like(grid, -1, dtype=np.int32)
        self.act_map = np.full((grid.shape[0], grid.shape[1], 4), -1, dtype=np.int32)  # ← 초기값 -1로 변경
        self.reachable_starts = reachable_starts
        self.height, self.width = grid.shape

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            "local":    gym.spaces.Box(0, 2, (3, 3), dtype=np.int32),
            "original": gym.spaces.Box(0, 2, self.grid.shape, dtype=np.int32),
            "memory":   gym.spaces.Box(-1, 3, self.grid.shape, dtype=np.int32),
            "act_mem":  gym.spaces.Box(-1, np.iinfo(np.int32).max, (self.height, self.width, 4), dtype=np.int32),
        })

    def _update_memory(self):
        y, x = self.agent_pos
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    self.memory_map[ny, nx] = self.grid[ny, nx]

    def reset(self, *, seed=None, options=None):
        self.agent_pos = list(random.choice(self.reachable_starts))
        self.steps = 0
        self.memory_map.fill(-1)
        self.act_map.fill(-1)  # 초기화 시 -1로 설정
        self._update_memory()
        return self._get_obs(), {}

    def _get_obs(self):
        y, x = self.agent_pos
        local = np.ones((3, 3), dtype=np.int32)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    local[dy + 1, dx + 1] = self.grid[ny, nx]
        return {
            "local": local,
            "original": self.original,
            "memory": self.memory_map.copy(),
            "act_mem": self.act_map.copy(),
        }

    def step(self, action):
        y, x = self.agent_pos
        dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        ny, nx = y + dy, x + dx

        moved = False
        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != 1:
            self.agent_pos = [ny, nx]
            moved = True

        self._update_memory()
        self.steps += 1
        self.act_map[y, x, action] = self.steps  # 현재 스텝 시점 기록

        terminated = self.agent_pos == list(self.goal)
        step_penalty = -0.01
        stay_penalty = -0.1 if not moved else 0.0
        reward = (10.0 if terminated else 0.0) + step_penalty + stay_penalty

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == 'human':
            view = np.array(self.grid, dtype=str)
            view[view == '0'] = '.'
            view[view == '1'] = '#'
            view[view == '2'] = 'G'
            y, x = self.agent_pos
            view[y, x] = 'A'
            print("\n".join("".join(row) for row in view))
            print()

    def close(self):
        pass

# ------------------------------------------------------------
# 사용 예시
# ------------------------------------------------------------

# # 1) 맵 준비 ---------------------------------------------------
# height, width, wall_prob = 15, 15, 0.7
# grid, goal, _ = generate_diverse_path(height, width, wall_prob)
# original = grid.copy()
# mutated  = mutate_walls_nearby(grid, 0.5, 3)

# reachable = [(y,x) for y in range(height) for x in range(width)
#              if mutated[y,x]==0 and is_path_exists(mutated,(y,x),goal)]
# env = GridEnv(mutated, goal, reachable, original)

# # 2) 랜덤 워크 --------------------------------------------------
# obs, _ = env.reset()
# max_step = 500          # 무한 루프 방지용 한도
# for step in range(1, max_step+1):
#     env.render()                     # 맵 출력
#     action = env.action_space.sample()
#     obs, _, terminated, truncated, _ = env.step(action)

#     # 🎯 목표 도달 여부는 reward 대신 terminated 플래그(or 좌표 비교)로 판단
#     if terminated:
#         print(f"🎉 Goal reached in {step} steps!")
#         break
# else:
#     print("❌ 최대 스텝 내에 goal 에 도달하지 못했습니다.")