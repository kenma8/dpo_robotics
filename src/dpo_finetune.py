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

TRAIN_SIZE = 250

def load_preferences(preferences_path):
    """Load preferences dataset from pickle file"""
    with open(preferences_path, "rb") as f:
        preferences = pickle.load(f)
    return preferences

def prepare_batch(preferences, batch_indices, device, seq_len=32):
    """
    Prepare a batch of preferences for training by sampling fixed-length segments
    from each trajectory
    """
    batch_chosen_obs = []
    batch_chosen_act = []
    batch_rejected_obs = []
    batch_rejected_act = []
    
    for i in batch_indices:
        # Get trajectory lengths
        chosen_len = len(preferences[i]["chosen_obs"])
        rejected_len = len(preferences[i]["rejected_obs"])
        
        # Sample start indices ensuring we have enough steps for seq_len
        chosen_start = np.random.randint(0, chosen_len - seq_len + 1)
        rejected_start = np.random.randint(0, rejected_len - seq_len + 1)
        
        # Extract segments
        chosen_obs = preferences[i]["chosen_obs"][chosen_start:chosen_start + seq_len]
        chosen_act = preferences[i]["chosen_act"][chosen_start:chosen_start + seq_len]
        rejected_obs = preferences[i]["rejected_obs"][rejected_start:rejected_start + seq_len]
        rejected_act = preferences[i]["rejected_act"][rejected_start:rejected_start + seq_len]
        
        batch_chosen_obs.append(chosen_obs)
        batch_chosen_act.append(chosen_act)
        batch_rejected_obs.append(rejected_obs)
        batch_rejected_act.append(rejected_act)
    
    # Stack into tensors
    batch_chosen_obs = torch.tensor(np.stack(batch_chosen_obs), dtype=torch.float32, device=device)
    batch_chosen_act = torch.tensor(np.stack(batch_chosen_act), dtype=torch.float32, device=device)
    batch_rejected_obs = torch.tensor(np.stack(batch_rejected_obs), dtype=torch.float32, device=device)
    batch_rejected_act = torch.tensor(np.stack(batch_rejected_act), dtype=torch.float32, device=device)
    
    return batch_chosen_obs, batch_chosen_act, batch_rejected_obs, batch_rejected_act

def get_sequence_log_probs(policy, obs_seq, act_seq):
    """Get log probs for a sequence using the RNN policy"""
    mu, std, _ = policy(obs_seq)  # (batch_size, seq_len, act_dim)
    dist = Independent(Normal(mu, std), 1)  # Make independent over action dimensions
    return dist.log_prob(act_seq)  # (batch_size, seq_len)

def dpo_loss(policy, old_policy, chosen_obs, chosen_act, rejected_obs, rejected_act, beta=0.1, kl_coef=0.1):
    """
    Compute DPO loss with KL regularization
    """
    # Get log probs for chosen and rejected trajectories
    chosen_log_probs = get_sequence_log_probs(policy, chosen_obs, chosen_act)
    rejected_log_probs = get_sequence_log_probs(policy, rejected_obs, rejected_act)
    
    # Sum log probs across time dimension
    chosen_log_probs = chosen_log_probs.sum(dim=1)  # [batch_size]
    rejected_log_probs = rejected_log_probs.sum(dim=1)  # [batch_size]
    
    # Compute DPO loss
    dpo = -torch.mean(
        torch.log(
            torch.sigmoid(beta * (chosen_log_probs - rejected_log_probs))
        )
    )
    
    # Add KL penalty
    with torch.no_grad():
        old_mu, old_std, _ = old_policy(chosen_obs)
    mu, std, _ = policy(chosen_obs)
    kl_div = torch.mean(torch.log(std/old_std) + (old_std**2 + (old_mu - mu)**2)/(2*std**2) - 0.5)
    
    return dpo + kl_coef * kl_div

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
    env = gym.make("Humanoid-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Load pretrained policy
    policy = BCPolicyRNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.model_path, map_location=args.device))
    policy = policy.to(args.device)
    policy.train()
    
    # Create copy of old policy for KL penalty
    old_policy = BCPolicyRNN(obs_dim, act_dim)
    old_policy.load_state_dict(policy.state_dict())
    old_policy = old_policy.to(args.device)
    old_policy.eval()
    
    # Load preferences dataset
    preferences = load_preferences(args.preferences_path)
    
    # Filter out preferences with trajectories shorter than seq_len
    seq_len = 32  # Same as used in training
    valid_preferences = []
    for pref in preferences:
        if (len(pref["chosen_obs"]) >= seq_len and 
            len(pref["rejected_obs"]) >= seq_len):
            valid_preferences.append(pref)
    
    preferences = valid_preferences[:TRAIN_SIZE]
    num_prefs = len(preferences)
    print(f"Using {num_prefs} preferences after filtering for minimum length")
    
    # Setup optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    
    # Add early stopping
    patience = 10
    no_improvement = 0
    
    # Add learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
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
            loss = dpo_loss(policy, old_policy,
                          batch_chosen_obs, batch_chosen_act,
                          batch_rejected_obs, batch_rejected_act,
                          beta=args.beta,
                          kl_coef=args.kl_coef)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate policy
        if (epoch + 1) % args.eval_interval == 0:
            mean_return = evaluate_policy(policy, env, args.device)
            print(f"Epoch {epoch+1}, Mean Return: {mean_return:.2f}")
            
            # Learning rate scheduling
            scheduler.step(mean_return)
            
            # Save best model and check early stopping
            if mean_return > best_return:
                best_return = mean_return
                torch.save(policy.state_dict(), args.save_path)
                print(f"Saved new best model with return {mean_return:.2f}")
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stopping triggered!")
                    break
    
    # Load best model at the end
    policy.load_state_dict(torch.load(args.save_path))

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
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Temperature parameter for DPO loss")
    parser.add_argument("--kl-coef", type=float, default=0.1,
                       help="KL divergence coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--eval-interval", type=int, default=5,
                       help="Epochs between evaluations")
    
    args = parser.parse_args()
    main(args)
