import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from wrapper import RandomMapWrapper
from env import GridEnv, generate_diverse_path, mutate_walls_nearby, is_path_exists
from net import MemoryExtractor
from VecEnvRenderWrapper import VecEnvRenderWrapper

def make_env(seed=None):
    env = RandomMapWrapper(15, 15, 0.3, 0, 3, render_mode='human')
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=200)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env

# 환경 설정 (seed 고정)
env = DummyVecEnv([lambda: make_env(seed=123)])
env = VecEnvRenderWrapper(env)

# 모델 불러오기
model = RecurrentPPO.load(
    "best/best_model",
    env=env,
    custom_objects={"features_extractor_class": MemoryExtractor}
)

# 초기 상태 설정
obs = env.reset()
state = None
episode_rewards = 0.0
done = [False]
step_num = 1

# 초기 환경
print("Initial env")
env.render()

printed = 0
while True:
    action, state = model.predict(obs, state=state, deterministic=False)
    obs, reward, done, info = env.step(action)

    if done[0]:
        print(f"Step: {step_num}, Action: {action[0]}, Reward: {reward[0]:.3f}, Done: {done[0]}, Info: {info[0]}")
        print(f"\n✅ Episode total reward: {episode_rewards:.3f}")
        printed = 1
        break

    print(f"Step: {step_num}, Action: {action[0]}, Reward: {reward[0]:.3f}, Done: {done[0]}, Info: {info[0]}")
    env.render()

    episode_rewards += reward[0]
    step_num += 1
    time.sleep(0.1)

if printed==0:
    print(f"\n✅ Episode total reward: {episode_rewards:.3f}")
