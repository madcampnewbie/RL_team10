from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from env import GridEnv 
# Define the same environment used in training
env = DummyVecEnv([lambda: GridEnv()])  # Use VecMonitor for monitoring

env = VecMonitor(env)  # Wrap the environment with VecMonitor for tracking episode statistics
# Load the model
model = RecurrentPPO.load("ppo_recurrent", env=env)

import numpy as np
state = None
obs = env.reset()
done = False
episode_start = True
i = 0
start_pos = env.envs[0].agent_pos
while not done:
    action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=False)
    obs, reward, done, info = env.step(action)
    episode_start = done
    i += 1

    if done or i >= 1000:
        print("Episode finished in {} steps.".format(i))
        print(env.envs[0].grid)
        print("Start position:", start_pos)
        print("Final position:", env.envs[0].agent_pos)
        break



