import numpy as np
import random
import gym
from gym import spaces
from collections import deque
import heapq

class GridEnv(gym.Env):
    '''
    Grid Environment for generalized pathfinding tasks \n
    '''
    def __init__(self,render_mode='human'):
        super(GridEnv, self).__init__()
        self.grid, self.goal, self.reachable_starts  = self.generate_diverse_path()
        self.grid = self.mutate_walls_nearby(mutation_rate=0.2, patch_size=3)

        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(
                                                low=-np.inf,
                                                high=np.inf,
                                                shape=(109,),
                                                dtype=np.float32
                                            )

    def reset(self):
        '''
        reset the environment to an initial state in possible random \n
        Returns: initial observation
        '''
        self.grid, self.goal, self.reachable_starts  = self.generate_diverse_path()
        self.grid = self.mutate_walls_nearby(mutation_rate=0.4, patch_size=5)

        self.agent_pos = list(random.choice(self.reachable_starts))
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        '''
        helper function to get the current observation \n
        Returns: 3x3 observation grid centered around the agent
        '''
        y, x = self.agent_pos
        obs = np.ones((3, 3), dtype=np.int32)  # 벽으로 초기화
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    obs[dy + 1, dx + 1] = self.grid[ny, nx]
        obs =  obs.flatten()
        # ret = np.concatenate([obs,self.compute_policy_field().flatten()]).astype(np.float32)
        ret = np.concatenate([obs,self.map_helper_function().flatten()]).astype(np.float32)
        return ret

    def step(self, action):
        y, x = self.agent_pos
        prev_pos = (y, x)
        dy, dx = [( -1, 0), (1, 0), (0, -1), (0, 1)][action] # down up left right
        ny, nx = y + dy, x + dx

        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != 1:
            self.agent_pos = [ny, nx]

        self.steps += 1
        done = self.agent_pos == list(self.goal)
        
        if prev_pos == self.agent_pos:
            reward = -1.1
        else:
            reward = 10 if done else -1
        
        self.grid = self.mutate_walls_nearby(mutation_rate=0.2, patch_size=3)

        
        return self._get_obs(), reward, done, {}

    def render(self,mode='human'):
        view = np.array(self.grid, dtype=str)
        view[view == '0'] = '.'
        view[view == '1'] = '#'
        view[view == '2'] = 'G'
        y, x = self.agent_pos
        view[y, x] = 'A'
        print('\n'.join(''.join(row) for row in view))
        print()

    def is_path_exists(self,start,goal):
        grid = self.grid

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


    def generate_diverse_path(self, map_size = (10,10),wall_prob=0.3):
        self.height, self.width = map_size
        height, width = self.height, self.width

        while True:
            self.grid = np.zeros((height, width), dtype=np.int32)
            grid = self.grid

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
                    if grid[y, x] == 0 and self.is_path_exists((y,x),goal):
                        reachable.append((y, x))

            # 4. 유효한 시작점이 충분하다면 맵 채택
            if len(reachable) > 0:
                return grid, goal, reachable


    def mutate_walls_nearby(self,mutation_rate=0.2, patch_size=3):
        '''
        A mutation function that randomly moves walls within the patch size \n
        Args:
            mutation_rate (float) [0,1]
            patch_size (int) n by n patch size
        Returns: mutated grid
        '''
        grid = self.grid 
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
    
    def compute_policy_field(self):
        grid = self.grid
        goal = self.goal

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        action_map = {
            (-1, 0): 1, #up
            (1, 0): 2, #down
            (0, -1): 3, #left
            (0, 1): 4, # right
            (0, 0): 0,  # goal
        }
        H, W = len(grid), len(grid[0])
        gy, gx = goal
        dist = [[float('inf')] * W for _ in range(H)]
        dist[gy][gx] = 0
        queue = [(0, gy, gx)]

        while queue:
            cost, y, x = heapq.heappop(queue)
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 0:
                    if dist[ny][nx] > cost + 1:
                        dist[ny][nx] = cost + 1
                        heapq.heappush(queue, (dist[ny][nx], ny, nx))

        # 각 셀에서 가장 가까운 방향으로의 액션 계산
        policy = np.zeros((H,W), dtype=int)
        for y in range(H):
            for x in range(W):
                if grid[y][x] == 1:
                    continue
                if (y, x) == goal:
                    policy[y][x] = 0
                    continue
                if dist[y][x] == float('inf'):
                    policy[y][x] = -1
                    continue

                min_cost = dist[y][x]
                best_action = (0, 0)
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] < min_cost:
                        min_cost = dist[ny][nx]
                        best_action = (dy, dx)

                policy[y][x] = action_map[best_action]

        return policy
    
    def map_helper_function(self):
        grid = self.grid
        goal = self.goal

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        action_map = {
            (-1, 0): 1, #up
            (1, 0): 2, #down
            (0, -1): 3, #left
            (0, 1): 4, # right
            (0, 0): 0,  # goal
        }
        H, W = len(grid), len(grid[0])
        gy, gx = goal
        dist = [[float('inf')] * W for _ in range(H)]
        dist[gy][gx] = 0
        queue = [(0, gy, gx)]

        while queue:
            cost, y, x = heapq.heappop(queue)
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 0:
                    if dist[ny][nx] > cost + 1:
                        dist[ny][nx] = cost + 1
                        heapq.heappush(queue, (dist[ny][nx], ny, nx))

        # 각 셀에서 가장 가까운 방향으로의 액션 계산
        policy = np.zeros((H,W), dtype=int)
        for y in range(H):
            for x in range(W):
                if grid[y][x] == 1:
                    continue
                if (y, x) == goal:
                    policy[y][x] = 2
                    continue
                if dist[y][x] == float('inf'):
                    policy[y][x] = -1
                    continue

                min_cost = dist[y][x]
                best_action = (0, 0)
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] < min_cost:
                        min_cost = dist[ny][nx]
                        best_action = (dy, dx)

                policy[y][x] = 0

        return policy
