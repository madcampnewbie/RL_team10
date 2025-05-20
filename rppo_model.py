import gym
from env import CustomEnv  # Assuming you have a custom environment defined in env.py
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList

# Register your environment or use an existing one
env = DummyVecEnv([lambda: CustomEnv()])

eval_env = DummyVecEnv([lambda: CustomEnv()])

eval_callback = EvalCallback(eval_env, eval_freq=1000, best_model_save_path="./logs/best")
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/checkpoints", name_prefix="recurrent_model")

callback = CallbackList([eval_callback, checkpoint_callback])

# Set random seed
set_random_seed(42)

# Initialize Recurrent PPO with LSTM policy
model = RecurrentPPO(
    policy=MlpLstmPolicy,
    env=env,
    verbose=1,
    n_steps=128,         # rollout length
    batch_size=64,
    n_epochs=4,
    learning_rate=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
)

# Train the model
model.learn(total_timesteps=10000,callback=callback)
