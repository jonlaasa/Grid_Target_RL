import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymtonic.envs

n_rows = 5
n_columns = 5

env = gym.make('gymtonic/GridTarget-v0', n_rows=n_rows, n_columns=n_columns, render_mode=None)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

env = gym.make('gymtonic/GridTarget-v0', n_rows=n_rows, n_columns=n_columns, render_mode='human')
env = Monitor(env)
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, render=True)
print(f"Mean_reward:{mean_reward:.2f}")