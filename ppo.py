# ppo.py
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wrapper import RandomMapWrapper
from gymnasium.wrappers import TimeLimit        # ★ 하드 타임아웃
from net import MemoryExtractor
from env import (
    GridEnv, generate_diverse_path,
    mutate_walls_nearby, is_path_exists
)

HEIGHT, WIDTH, WALL_P = 7, 7, 0.3
MUT_RATE, PATCH = 0, 3
MAX_STEPS = 4000

# ---------------------------------------------------------------------------
def make_env(seed: int | None = None):
    env = RandomMapWrapper(
        height=HEIGHT,
        width=WIDTH,
        wall_prob=WALL_P,
        mut_rate=MUT_RATE,
        patch_size=PATCH
    )
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    N_ENVS = 4
    n_steps = 64
    batch_size = N_ENVS * n_steps

    # 1) 학습용 VecEnv 
    env = SubprocVecEnv([lambda i=i: make_env(seed=i) for i in range(N_ENVS)])

    # 2) Policy + 추출기 설정
    policy_kwargs = dict(
        features_extractor_class=MemoryExtractor,
        lstm_hidden_size=256,
    )

    # 3) 모델
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        ent_coef=0.05,
        gamma=0.99,
        tensorboard_log="./tb",
        verbose=1,
    )

    # 4-a) 체크포인트
    checkpoint_cb = CheckpointCallback(
        save_freq=2_000, save_path="./checkpoints", name_prefix="grid_agent"
    )

    # 4-b) 평가 콜백 (병렬 4개 환경, 8 에피소드 평가)
    eval_env = SubprocVecEnv([lambda i=i: make_env(seed=456 + i) for i in range(N_ENVS)])
    eval_freq = 1_000
    eval_cb  = EvalCallback(
        eval_env, best_model_save_path="./best",
        eval_freq=eval_freq//N_ENVS, n_eval_episodes=8,
        deterministic=False, render=False,
    )

    # 5) 학습
    try:
        model.learn(
            total_timesteps=100_000,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
            log_interval=1,
        )
    finally:
        model.save("grid_lstm_agent_last")     # 중단 시 마지막 상태 저장

    # 6) 최종 모델 저장
    model.save("grid_lstm_agent_final")
