import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from wrapper import RandomMapWrapper, StallTerminationWrapper  # 🔁 모든 래퍼
from gymnasium.wrappers import TimeLimit
from net import MemoryExtractor

# Grid 생성 설정
HEIGHT, WIDTH = 15, 15
WALL_P = 0.3
MUT_RATE = 0.01
PATCH = 3
MAX_STEPS = 300
K_STALL = 20

# 🧩 학습과 동일한 래퍼 체인 구성
def make_env():
    env = RandomMapWrapper(HEIGHT, WIDTH, WALL_P, MUT_RATE, PATCH)
    env = StallTerminationWrapper(env, k=K_STALL, max_steps=MAX_STEPS, step_penalty=-0.01)
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    return env

# DummyVecEnv 래핑
env = DummyVecEnv([make_env])

# 모델 로드
model = RecurrentPPO.load("./best/best_model.zip", env=env)

# 테스트 에피소드
obs = env.reset()
lstm_state = None
done = [False]
step_count = 0

while not done[0]:
    step_count += 1
    action, lstm_state = model.predict(obs, state=lstm_state, deterministic=False)
    obs, rewards, done, info = env.step(action)
    env.envs[0].render()

# 결과 출력
if rewards[0] > 0:
    print(f"🎉 Goal reached in {step_count} steps!")
else:
    print(f"❌ Failed to reach goal in {step_count} steps.")

env.close()
