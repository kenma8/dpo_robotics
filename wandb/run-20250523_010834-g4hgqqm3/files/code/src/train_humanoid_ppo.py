import gym
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
from wandb.integration.sb3 import WandbCallback

env_id = "Humanoid-v5"  
num_envs = 32

env = make_vec_env(env_id, n_envs=num_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
wandb.init(
    name=f"PPO_{timestamp}", 
    sync_tensorboard=True,  
    monitor_gym=True,       
    save_code=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PPO(
    "MlpPolicy",        
    env,
    device=device,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,        
    batch_size=256,
    n_epochs=1000,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])]),
    tensorboard_log=f"./runs/ppo_humanoid/{timestamp}"
)

total_timesteps = 655360
model.learn(
    total_timesteps=total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"./runs/ppo_humanoid/{timestamp}",
        verbose=2
    )
)

model.save("ppo_humanoid")
env.save("vec_normalize.pkl")

eval_env = make_vec_env(env_id, n_envs=1)
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")

wandb.finish()
