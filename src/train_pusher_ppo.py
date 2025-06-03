import gym
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env_id = "Pusher-v5"  
num_envs = 4

env = make_vec_env(env_id, n_envs=num_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PPO(
    "MlpPolicy",        
    env,
    device=device,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,        
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=f"./runs/ppo_pusher/{timestamp}"
)

total_timesteps = 10000000
model.learn(
    total_timesteps=total_timesteps,
)

model.save("./policies/ppo_pusher/ppo_pusher")
env.save("./policies/ppo_pusher/vec_normalize.pkl")

eval_env = make_vec_env(env_id, n_envs=1)
eval_env = VecNormalize.load("./policies/ppo_pusher/vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")

