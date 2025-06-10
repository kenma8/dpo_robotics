import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
from pathlib import Path
import gymnasium as gym
from train_humanoid_baseline import BCPolicyRNN
from torch.distributions import Normal, Independent
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def prepare_batch(preferences, batch_indices, device):
    """Prepare a batch of padded preference sequences"""
    def get_seq(tensor_list):
        # Convert to float32 tensors and pad
        tensors = [torch.tensor(t, dtype=torch.float32) for t in tensor_list]
        padded = pad_sequence(tensors, batch_first=True)
        return padded.to(device)

    chosen_obs = get_seq([preferences[i]["chosen_obs"] for i in batch_indices])
    chosen_act = get_seq([preferences[i]["chosen_act"] for i in batch_indices])
    rejected_obs = get_seq([preferences[i]["rejected_obs"] for i in batch_indices])
    rejected_act = get_seq([preferences[i]["rejected_act"] for i in batch_indices])

    return chosen_obs, chosen_act, rejected_obs, rejected_act


def build_mask(padded_seq):
    """Build a mask for padded sequences
    Args:
        padded_seq: [B, T, D] tensor of padded sequences
    Returns:
        [B, T] tensor with 1s for real values and 0s for padding
    """
    # Mask where any dimension in the feature vector is non-zero
    return (padded_seq.abs().sum(dim=-1) > 1e-6).float()  # [B, T]


def get_sequence_log_probs(policy, obs_seq, act_seq, mask=None):
    """Get log probabilities for each action in the sequence
    Args:
        policy: Policy network
        obs_seq: [B, T, obs_dim] tensor of observations
        act_seq: [B, T, act_dim] tensor of actions
        mask: [B, T] tensor indicating valid timesteps
    Returns:
        [B] tensor of sequence log probabilities
    """
    # Get predictions for entire sequence
    mu, std, _ = policy(obs_seq)  # [B, T, act_dim]
    
    # Compute log probs
    dist = Independent(Normal(mu, std), 1)
    timestep_log_probs = dist.log_prob(act_seq)  # [B, T]

    if mask is None:
        mask = torch.ones_like(timestep_log_probs)
    
    # Apply mask and average over valid timesteps
    masked_log_probs = timestep_log_probs * mask
    sequence_log_probs = masked_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    
    return sequence_log_probs  # [B]


def dpo_loss(policy, chosen_obs, chosen_act, rejected_obs, rejected_act, beta=0.1):
    """Compute DPO loss for a batch of preferences
    Args:
        policy: Policy network
        chosen_obs: [B, T, obs_dim] tensor of chosen observations
        chosen_act: [B, T, act_dim] tensor of chosen actions
        rejected_obs: [B, T, obs_dim] tensor of rejected observations
        rejected_act: [B, T, act_dim] tensor of rejected actions
        beta: Temperature parameter for DPO
    Returns:
        Scalar loss value
    """
    # Build masks for valid timesteps
    mask_chosen = build_mask(chosen_act)    # [B, T]
    mask_rejected = build_mask(rejected_act) # [B, T]
    
    # Get sequence log probs
    chosen_log_probs = get_sequence_log_probs(policy, chosen_obs, chosen_act, mask_chosen)   # [B]
    rejected_log_probs = get_sequence_log_probs(policy, rejected_obs, rejected_act, mask_rejected) # [B]

    # Compute DPO loss
    loss = -torch.mean(torch.log(torch.sigmoid(beta * (chosen_log_probs - rejected_log_probs))))
    return loss


def evaluate_policy(policy, env, device, num_episodes=10):
    """Evaluate policy performance"""
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_ret = 0.0
        obs_history = []  # Keep track of observation history
        hidden = None

        while not done:
            obs_tensor = torch.from_numpy(obs[None, None]).float().to(device)  # [1, 1, obs_dim]
            with torch.no_grad():
                mu, std, hidden = policy(obs_tensor, hidden)
            action = mu[0, 0].cpu().numpy()  # Take first timestep's action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_ret += reward
        returns.append(total_ret)

    return np.mean(returns)


def compute_reward_differences(dataset):
    """Compute absolute reward differences for each preference pair"""
    differences = []
    for pref in dataset:
        # Get rewards from metadata
        if "metadata" in pref:
            chosen_reward = pref["metadata"]["reward_chosen"]
            rejected_reward = pref["metadata"]["reward_rejected"]
        
        differences.append(abs(chosen_reward - rejected_reward))
    return np.array(differences)


def select_examples(dataset, num_examples, strategy='random'):
    """
    Select examples from dataset using specified strategy
    
    Args:
        dataset: List of preference pairs
        num_examples: Number of examples to select
        strategy: One of ['random', 'importance']
            - random: Uniform random sampling
            - importance: Take top k examples by reward difference
    """
    if num_examples >= len(dataset):
        print(f"Warning: requested {num_examples} examples but dataset only has {len(dataset)}. Using all examples.")
        return dataset
    
    if strategy == 'random':
        indices = np.random.permutation(len(dataset))[:num_examples]
        print(f"Using {num_examples} randomly sampled examples")
    
    elif strategy == 'importance':
        # Compute reward differences
        differences = compute_reward_differences(dataset)
        
        # Get indices of top k examples by reward difference
        indices = np.argsort(differences)[-num_examples:]
        print(f"Using top {num_examples} examples by reward difference")
        
        # Print statistics about selected examples
        selected_diffs = differences[indices]
        print(f"Selected examples reward diff stats:")
        print(f"Mean: {selected_diffs.mean():.2f}")
        print(f"Std: {selected_diffs.std():.2f}")
        print(f"Min: {selected_diffs.min():.2f}")
        print(f"Max: {selected_diffs.max():.2f}")
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return [dataset[i] for i in indices]


def train_dpo(model_path, preferences_path, num_examples=None, sampling_strategy=None,
               num_epochs=100, batch_size=32, learning_rate=5e-6, beta=0.05, device="cuda"):
    """Train a policy using DPO with the specified parameters"""
    
    # Setup environment and policy
    env = gym.make("Humanoid-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Load pretrained policy
    policy = BCPolicyRNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    policy = policy.to(device)
    policy.train()
    
    # Load dataset
    with open(preferences_path, 'rb') as f:
        full_dataset = pickle.load(f)
    
    if num_examples is not None:
        if sampling_strategy is not None:
            dataset = select_examples(full_dataset, num_examples, strategy=sampling_strategy)
        else:
            print(f"Using first {num_examples} examples from dataset (no active learning)")
            dataset = full_dataset[:num_examples]
    else:
        dataset = full_dataset
    
    num_prefs = len(dataset)
    print(f"Training on {num_prefs} preferences")
    
    # Adjust batch size if dataset is small
    batch_size = min(batch_size, num_prefs)
    num_batches = max(num_prefs // batch_size, 1)  # Ensure at least one batch
    print(f"Using batch size: {batch_size}, num batches: {num_batches}")
    
    # Setup optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training loop
    best_return = float('-inf')
    best_policy_state = None
    
    # Initial evaluation
    policy.eval()
    mean_return = evaluate_policy(policy, env, device)
    policy.train()
    best_return = mean_return
    best_policy_state = {k: v.cpu() for k, v in policy.state_dict().items()}
    print(f"Initial evaluation - Mean Return: {mean_return:.2f}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        perm = np.random.permutation(num_prefs)
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_prefs)  # Ensure we don't go past dataset size
            batch_indices = perm[start_idx:end_idx]
            
            # Prepare batch
            batch_chosen_obs, batch_chosen_act, batch_rejected_obs, batch_rejected_act = \
                prepare_batch(dataset, batch_indices, device)
            
            # Compute loss and update
            optimizer.zero_grad()
            loss = dpo_loss(policy,
                          batch_chosen_obs, batch_chosen_act,
                          batch_rejected_obs, batch_rejected_act,
                          beta=beta)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Print average loss
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            policy.eval()
            mean_return = evaluate_policy(policy, env, device)
            policy.train()
            
            print(f"Epoch {epoch + 1}, Mean Return: {mean_return:.2f}")
            
            if mean_return > best_return:
                print(f"New best return! {best_return:.2f} -> {mean_return:.2f}")
                best_return = mean_return
                best_policy_state = {k: v.cpu() for k, v in policy.state_dict().items()}
    
    print(f"\nTraining complete. Best return achieved: {best_return:.2f}")
    return best_return


def run_size_progression(model_path, preferences_path, sizes, strategies=['random', 'importance'],
                        num_seeds=3, device="cuda"):
    """Run experiments with increasing dataset sizes for each strategy"""
    
    results = {strategy: {size: [] for size in sizes} for strategy in strategies}
    
    for strategy in strategies:
        strategy_name = strategy if strategy else "no_active_learning"
        print(f"\nRunning experiments for strategy: {strategy_name}")
        
        for size in sizes:
            print(f"\nDataset size: {size}")
            
            # Run multiple seeds
            for seed in range(num_seeds):
                print(f"Seed {seed + 1}/{num_seeds}")
                
                # Set random seeds
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                final_return = train_dpo(
                    model_path=model_path,
                    preferences_path=preferences_path,
                    num_examples=size,
                    sampling_strategy=strategy,
                    device=device
                )
                
                results[strategy][size].append(final_return)
                
                # Save intermediate results after each experiment
                os.makedirs('plots', exist_ok=True)
                with open('plots/humanoid_intermediate_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
    
    return results


def plot_results(results, sizes, output_dir="plots"):
    """Create plots from experimental results"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for strategy, size_results in results.items():
        strategy_name = strategy if strategy else "No Active Learning"
        
        # Calculate mean and std across seeds
        means = [np.mean(size_results[size]) for size in sizes]
        stds = [np.std(size_results[size]) for size in sizes]
        
        # Plot with error bars
        plt.errorbar(sizes, means, yerr=stds, label=strategy_name, marker='o', capsize=5)
    
    plt.xscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Mean Return')
    plt.title('DPO Performance vs Dataset Size (Humanoid)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'humanoid_active_learning_comparison.png'))
    plt.close()
    
    # Save final results
    with open(os.path.join(output_dir, 'humanoid_final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def main():
    # Configuration
    model_path = "policies/bcc_rnn_humanoid_probabilistic/bc_rnn_humanoid.pth"
    preferences_path = "src/collect_human_preferences/preferences/generated_combined.pkl"
    
    sizes = [5, 10, 25, 50, 100, 500, 1000, 2000]
    strategies = ['importance', 'random']
    
    # Run experiments
    results = run_size_progression(
        model_path=model_path,
        preferences_path=preferences_path,
        sizes=sizes,
        strategies=strategies,
        num_seeds=3,  # Number of random seeds to run for each configuration
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create plots
    plot_results(results, sizes)


if __name__ == "__main__":
    main() 