import torch
import gymnasium as gym
import numpy as np
import argparse
from train_humanoid_baseline import BCPolicyRNN
from train_pusher_baseline import BCPolicyMLP
from multiprocessing import Pool
from functools import partial
import os

MODEL_REGISTRY = {
    "BCPolicyRNN": BCPolicyRNN,
    "BCPolicyMLP": BCPolicyMLP,
    # Add more models as needed
}

def evaluate_episode(env_name, policy, device, seed, render=False):
    """Evaluate a single episode with a given seed"""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    done = False
    total_ret = 0.0
    hidden = None
    
    while not done:
        obs_tensor = torch.from_numpy(obs[None, None]).float().to(device)
        with torch.no_grad():
            if isinstance(policy, BCPolicyRNN):
                mu, std, hidden = policy(obs_tensor, hidden)
            else:  # BCPolicyMLP
                mu, std, _ = policy(obs_tensor)
        # Use deterministic mean at test time
        action = mu[0, 0].cpu().numpy()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        if render:
            env.render()
        
        done = terminated or truncated
        total_ret += reward
    
    print(f"finished seed: {seed}")
    
    return seed, total_ret

def evaluate_policy(policy, env_name, device, num_episodes=10, render=False, num_workers=None):
    """Evaluate policy performance using parallel workers"""
    if num_workers is None:
        num_workers = min(os.cpu_count(), num_episodes)
    
    # Create partial function with fixed arguments
    eval_fn = partial(evaluate_episode, env_name, policy, device, render=render)
    
    # Generate seeds for reproducibility
    seeds = list(range(num_episodes))
    
    returns_dict = {}
    if num_workers > 1:
        with Pool(num_workers) as pool:
            # Map seeds to evaluation function
            results = pool.map(eval_fn, seeds)
            for seed, ret in results:
                returns_dict[seed] = ret
                print(f"Episode {seed+1}: Return = {ret:.2f}")
    else:
        # Single process evaluation
        for seed in seeds:
            _, ret = eval_fn(seed)
            returns_dict[seed] = ret
            print(f"Episode {seed+1}: Return = {ret:.2f}")
    
    # Sort returns by seed to maintain deterministic order
    returns = [returns_dict[seed] for seed in sorted(returns_dict.keys())]
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min return: {min(returns):.2f}")
    print(f"Max return: {max(returns):.2f}")
    
    return mean_return, std_return

def main(args):
    # Setup environment (just for dimensions)
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()
    
    # Load policy
    if args.model_class not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class: {args.model_class}. Must be one of: {list(MODEL_REGISTRY.keys())}")
    
    policy_class = MODEL_REGISTRY[args.model_class]
    policy = policy_class(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.model_path, map_location=args.device))
    policy = policy.to(args.device)
    policy.eval()
    
    # Evaluate
    evaluate_policy(policy, args.env_name, args.device, args.num_episodes, args.render, args.num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model weights")
    parser.add_argument("--env-name", type=str, required=True,
                       help="Name of the environment (e.g., Humanoid-v4, Pusher-v4)")
    parser.add_argument("--model-class", type=str, required=True,
                       help="Model class name (BCPolicyRNN or BCPolicyMLP)")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers. Defaults to min(cpu_count, num_episodes)")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    
    args = parser.parse_args()
    main(args) 