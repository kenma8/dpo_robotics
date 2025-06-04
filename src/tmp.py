import os
import glob
import torch
import gymnasium as gym
import numpy as np
import argparse
from train_pusher_baseline import BCPolicyMLP  # adjust if import path differs


def evaluate_policy(policy, env, device, num_episodes=10, render=False):
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_ret = 0.0

        while not done:
            obs_tensor = torch.from_numpy(obs[None]).float().to(device)
            with torch.no_grad():
                mu, std, _ = policy(obs_tensor)
            action = mu[0].cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            done = terminated or truncated
            total_ret += reward

        returns.append(total_ret)


    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min return: {min(returns):.2f}")
    print(f"Max return: {max(returns):.2f}")
    return mean_return, std_return


def tmp():
    env = gym.make("Pusher-v5", render_mode="human" if False else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_files = sorted(glob.glob(os.path.join("/Users/21stewartp/Desktop/StanfordWork/CS224R/FinalProject/dpo_robotics/policies/dpo_pusher_human", "*.pth")))
    if not model_files:
        print(f"No .pth files found in /Users/21stewartp/Desktop/StanfordWork/CS224R/FinalProject/dpo_robotics/policies/dpo_pusher_human")
        return
    print("Found model files:", model_files)

    for model_path in model_files:
        print(f"Evaluating {os.path.basename(model_path)}")
        model = BCPolicyMLP(obs_dim, act_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to("cpu")
        model.eval()

        mean, std = evaluate_policy(model, env, "cpu", 1000, False)
        print(f"→ {os.path.basename(model_path)} → mean: {mean:.2f}, std: {std:.2f}\n")


tmp()