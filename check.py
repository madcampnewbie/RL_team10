from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from ppo import make_env

# Monitor → DummyVecEnv 래핑
env = DummyVecEnv([lambda: Monitor(make_env())])

# 학습한 모델 로드
model = RecurrentPPO.load("grid_lstm_agent_final.zip", env=env)
print("Model is loaded")

# 20에피소드 성능 평가
mean_rew, std_rew = evaluate_policy(
    model, env,
    n_eval_episodes=20,
    deterministic=True           # True: 완전 결정적 정책
)

print(f"평균 보상: {mean_rew:.2f} ± {std_rew:.2f}")
