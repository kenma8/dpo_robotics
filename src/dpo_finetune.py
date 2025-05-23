import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from tqdm import tqdm
import os
import gymnasium as gym
from torch.distributions import Normal, Independent

from train_humanoid_baseline import BCPolicyRNN

def load_preferences(preferences_path):
    """Load preferences dataset from pickle file"""
    with open(preferences_path, "rb") as f:
        preferences = pickle.load(f)
    return preferences

def prepare_batch(preferences, batch_indices, device):
    """Prepare a batch of preferences for training"""
    batch_chosen_obs = torch.tensor(np.stack([preferences[i]["chosen_obs"] for i in batch_indices]), 
                                  dtype=torch.float32, device=device)
    batch_chosen_act = torch.tensor(np.stack([preferences[i]["chosen_act"] for i in batch_indices]), 
                                  dtype=torch.float32, device=device)
    batch_rejected_obs = torch.tensor(np.stack([preferences[i]["rejected_obs"] for i in batch_indices]), 
                                    dtype=torch.float32, device=device)
    batch_rejected_act = torch.tensor(np.stack([preferences[i]["rejected_act"] for i in batch_indices]), 
                                    dtype=torch.float32, device=device)
    
    return batch_chosen_obs, batch_chosen_act, batch_rejected_obs, batch_rejected_act

def get_sequence_log_probs(policy, obs_seq, act_seq):
    """Get log probs for a sequence using the RNN policy"""
    mu, std, _ = policy(obs_seq)  # (batch_size, seq_len, act_dim)
    dist = Independent(Normal(mu, std), 1)  # Make independent over action dimensions
    return dist.log_prob(act_seq)  # (batch_size, seq_len)

def dpo_loss(policy, chosen_obs, chosen_act, rejected_obs, rejected_act, beta=0.1):
    """
    Compute DPO loss for a batch of preferences
    
    Args:
        policy: BCPolicyRNN instance
        chosen_obs: Observations from chosen trajectories [batch_size, seq_len, obs_dim]
        chosen_act: Actions from chosen trajectories [batch_size, seq_len, act_dim]
        rejected_obs: Observations from rejected trajectories [batch_size, seq_len, obs_dim]
        rejected_act: Actions from rejected trajectories [batch_size, seq_len, act_dim]
        beta: Temperature parameter for DPO loss
    """
    # Get log probs for chosen and rejected trajectories
    chosen_log_probs = get_sequence_log_probs(policy, chosen_obs, chosen_act)
    rejected_log_probs = get_sequence_log_probs(policy, rejected_obs, rejected_act)
    
    # Sum log probs across time dimension
    chosen_log_probs = chosen_log_probs.sum(dim=1)  # [batch_size]
    rejected_log_probs = rejected_log_probs.sum(dim=1)  # [batch_size]
    
    # Compute DPO loss
    loss = -torch.mean(
        torch.log(
            torch.sigmoid(beta * (chosen_log_probs - rejected_log_probs))
        )
    )
    
    return loss

def evaluate_policy(policy, env, device, num_episodes=10):
    """Evaluate policy performance"""
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
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
            done = terminated or truncated
            total_ret += reward
            
        returns.append(total_ret)
    
    return np.mean(returns)

def main(args):
    # Setup environment
    env = gym.make("Humanoid-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Load pretrained policy
    policy = BCPolicyRNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.model_path, map_location=args.device))
    policy = policy.to(args.device)
    policy.train()
    
    # Load preferences dataset
    preferences = load_preferences(args.preferences_path)
    num_prefs = len(preferences)
    print(f"Loaded {num_prefs} preferences")
    
    # Setup optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    
    # Training loop
    batch_size = args.batch_size
    num_batches = num_prefs // batch_size
    
    best_return = float('-inf')
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        
        # Shuffle preferences
        perm = np.random.permutation(num_prefs)
        
        # Train on batches
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            start_idx = batch_idx * batch_size
            batch_indices = perm[start_idx:start_idx + batch_size]
            
            # Prepare batch
            batch_chosen_obs, batch_chosen_act, batch_rejected_obs, batch_rejected_act = \
                prepare_batch(preferences, batch_indices, args.device)
            
            # Compute loss and update
            optimizer.zero_grad()
            loss = dpo_loss(policy, 
                          batch_chosen_obs, batch_chosen_act,
                          batch_rejected_obs, batch_rejected_act,
                          beta=args.beta)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate policy
        if (epoch + 1) % args.eval_interval == 0:
            mean_return = evaluate_policy(policy, env, args.device)
            print(f"Epoch {epoch+1}, Mean Return: {mean_return:.2f}")
            
            # Save best model based on returns
            if mean_return > best_return:
                best_return = mean_return
                torch.save(policy.state_dict(), args.save_path)
                print(f"Saved new best model with return {mean_return:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to pretrained BCPolicyRNN model")
    parser.add_argument("--preferences-path", type=str, required=True,
                       help="Path to preferences dataset")
    parser.add_argument("--save-path", type=str, required=True,
                       help="Path to save finetuned model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Temperature parameter for DPO loss")
    parser.add_argument("--eval-interval", type=int, default=5,
                       help="Epochs between evaluations")
    
    args = parser.parse_args()
    main(args)
