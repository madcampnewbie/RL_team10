import gymnasium as gym
from typing import List, Tuple

# ------------------------------------------------------------
# 매 에피소드마다 맵을 재생성하는 래퍼
# ------------------------------------------------------------
class RandomMapWrapper(gym.Wrapper):
    def __init__(
        self, height: int, width: int, wall_prob: float, mut_rate: float, patch_size: int, render_mode=None
    ):
        from env import generate_diverse_path, mutate_walls_nearby, GridEnv, is_path_exists

        self.height = height
        self.width = width
        self.wall_prob = wall_prob
        self.mut_rate = mut_rate
        self.patch_size = patch_size
        self._render_mode = render_mode  # 변수명 변경 필수

        while True:
            grid, goal, reachable = generate_diverse_path(height, width, wall_prob)
            mutated = mutate_walls_nearby(grid, mut_rate, patch_size)
            reachable = [
                (y, x)
                for y in range(height)
                for x in range(width)
                if mutated[y, x] == 0 and is_path_exists(mutated, (y, x), goal)
            ]
            if reachable:
                break

        dummy = GridEnv(mutated, goal, reachable, grid.copy(), render_mode=self._render_mode)
        super().__init__(dummy)

    def reset(self, **kwargs):
        from env import generate_diverse_path, mutate_walls_nearby, is_path_exists, GridEnv

        while True:
            grid, goal, _ = generate_diverse_path(self.height, self.width, self.wall_prob)
            mutated = mutate_walls_nearby(grid, self.mut_rate, self.patch_size)
            reachable = [
                (y, x)
                for y in range(self.height)
                for x in range(self.width)
                if mutated[y, x] == 0 and is_path_exists(mutated, (y, x), goal)
            ]
            if reachable:
                break

        self.env = GridEnv(mutated, goal, reachable, grid.copy(), render_mode=self._render_mode)

        return self.env.reset(**kwargs)

    @property
    def render_mode(self):
        return self._render_mode




# # ------------------------------------------------------------
# # 최근 k 스텝 동안 위치 변화가 없으면 종료
# # ------------------------------------------------------------
# class StallTerminationWrapper(gym.Wrapper):
#     """
#     최근 k 스텝 동안 위치 변화가 없으면 truncated=True 로 종료하고,
#     남은 스텝(step_penalty × 남은 스텝 수)만큼 보상을 한꺼번에 감산한다.
#     """
#     def __init__(
#         self,
#         env: gym.Env,
#         k: int = 20,
#         max_steps: int = 200,
#         step_penalty: float = -0.01,
#     ):
#         super().__init__(env)
#         self.k = k
#         self.max_steps = max_steps
#         self.step_penalty = step_penalty
#         self.hist: List[Tuple[int, int]] = []
#         self._local_step = 0  # 스텝 카운터

#     def _pos(self) -> Tuple[int, int]:
#         return tuple(self.env.unwrapped.agent_pos)

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.hist = [self._pos()]
#         self._local_step = 0
#         return obs, info

#     def step(self, action):
#         obs, rew, term, trunc, info = self.env.step(action)
#         self._local_step += 1

#         # k 스텝 동안 제자리 → stall
#         self.hist.append(self._pos())
#         if len(self.hist) > self.k:
#             self.hist.pop(0)
#             if len(set(self.hist)) == 1 and not term:
#                 term, trunc = True, False
#                 info["truncated_by"] = "stall"

#         # truncated 시 남은 스텝 패널티
#         if trunc and not term:
#             remaining = self.max_steps - self._local_step
#             rew += self.step_penalty * max(remaining, 0)

#         return obs, rew, term, trunc, info
