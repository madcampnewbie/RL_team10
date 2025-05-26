from stable_baselines3.common.vec_env import VecEnvWrapper

class VecEnvRenderWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def render(self):
        return self.venv.envs[0].render()

