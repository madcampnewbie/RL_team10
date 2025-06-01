import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
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


def generate_diverse_path(height, width, wall_prob, seed = None):
    if seed is not None:
        random.seed(seed)

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
def mutate_walls_nearby(grid, goal, mutation_rate=0.2, patch_size=3, seed = None):
    if seed is not None:
        random.seed(seed)
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

    reachable = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if mutated[y, x] == 0 and is_path_exists(mutated, (y, x), goal):
                reachable.append((y, x))

    return mutated, reachable

# ------------------------------------------------------------
# 강화학습 환경 클래스
# ------------------------------------------------------------
class GridEnv(gym.Env):
    def __init__(self, grid, goal, reachable_starts):
        super(GridEnv, self).__init__()
        self.grid = grid
        self.goal = goal
        self.reachable_starts = reachable_starts
        self.height, self.width = grid.shape

        self.action_space = spaces.Discrete(4)  # 상하좌우
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int32) # 관찰 가능한 공간 (3x3)

    def reset(self):
        # reachable 한 빈 칸 중 하나에서 agent 배치
        self.agent_pos = list(random.choice(self.reachable_starts))
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        y, x = self.agent_pos
        obs = np.ones((3, 3), dtype=np.int32)  # 벽으로 초기화
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    obs[dy + 1, dx + 1] = self.grid[ny, nx]
        return obs

    def step(self, action):
        y, x = self.agent_pos
        dy, dx = [( -1, 0), (1, 0), (0, -1), (0, 1)][action]
        ny, nx = y + dy, x + dx

        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != 1:
            self.agent_pos = [ny, nx]

        self.steps += 1
        done = self.agent_pos == list(self.goal)
        reward = 1 if done else 0
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        view = np.array(self.grid, dtype=str)
        view[view == '0'] = '.'
        view[view == '1'] = '#'
        view[view == '2'] = 'G'
        y, x = self.agent_pos
        view[y, x] = 'A'
        print('\n'.join(''.join(row) for row in view))
        print()

    def close(self):
        pass

# ------------------------------------------------------------
# 사용 예시
# ------------------------------------------------------------


def main():
    # 원본 맵 생성 (goal과 reachable start 리스트 포함)
    grid, goal, reachable_starts = generate_diverse_path(height=10, width=10, wall_prob=0.2)

    # 변형 적용
    mutated_grid = mutate_walls_nearby(grid, mutation_rate=0.2, patch_size=3)

    # 환경 생성 (변형된 맵 + 기존 goal 위치 + reachable 시작점)
    env = GridEnv(mutated_grid, goal, reachable_starts)

    # 환경 초기화
    obs = env.reset()
    done = False

    # 랜덤한 정책으로 실행
    while not done:
        env.render()
        action = env.action_space.sample()  # 무작위 행동
        obs, reward, done, _ = env.step(action)

    if reward == 1:
        print("🎉 Goal reached!")
    else:
        print("❌ Failed to reach the goal.")

if __name__ == "__main__":
    main()