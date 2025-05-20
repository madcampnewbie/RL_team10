from sb3_contrib import RecurrentA2C
from stable_baselines3.common.envs import DummyVecEnv
from env import CustomEnv
import gym

env = DummyVecEnv([lambda: CustomEnv()])

model = RecurrentA2C("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
