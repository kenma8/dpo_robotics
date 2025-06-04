import torch
import gymnasium as gym
import numpy as np
import argparse
from train_humanoid_baseline import BCPolicyRNN

def evaluate_policy(policy, env, device, num_episodes=10, render=False):
    """Evaluate policy performance"""
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_ret = 0.0
        hidden = None
        
        while not done:
            obs_tensor = torch.from_numpy(obs[None, None]).float().to(device)
            with torch.no_grad():
                mu, std, hidden = policy(obs_tensor, hidden)
            # Use deterministic mean at test time
            action = mu[0, 0].cpu().numpy()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            
            done = terminated or truncated
            total_ret += reward
            
        returns.append(total_ret)
        print(f"Episode {ep+1}: Return = {total_ret:.2f}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min return: {min(returns):.2f}")
    print(f"Max return: {max(returns):.2f}")
    
    return mean_return, std_return

def main(args):
    # Setup environment
    env = gym.make("Humanoid-v5", render_mode="human" if args.render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Load policy
    policy = BCPolicyRNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.model_path, map_location=args.device))
    policy = policy.to(args.device)
    policy.eval()
    
    # Evaluate
    evaluate_policy(policy, env, args.device, args.num_episodes, args.render)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model weights")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    
    args = parser.parse_args()
    main(args) 