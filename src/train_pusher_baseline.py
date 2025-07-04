import gymnasium as gym
import minari                   # pip install minari[all]
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from torch.distributions import Normal, Independent

SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-3
TRAIN_ROllOUTS = 50
EVAL_ROLLOUTS = 200
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 2

if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    
print(f"Using device: {DEVICE}")

MODEL_PATH = "policies/bc_mlp_pusher/bc_mlp_pusher_3.pth"
OPT_PATH = "policies/bc_mlp_pusher/bc_mlp_pusher_opt_3.pth"

class SequenceDataset(Dataset):
    def __init__(self, episodes, seq_len):
        self.episodes = episodes
        self.seq_len = seq_len

    def __len__(self):
        return sum(max(0, ep['observations'].shape[0] - self.seq_len + 1)
                for ep in self.episodes)

    def __getitem__(self, idx):
        cum = 0
        for ep in self.episodes:
            L = ep['observations'].shape[0] - self.seq_len + 1
            if idx < cum + max(0, L):
                start = idx - cum
                end   = start + self.seq_len
                o_seq = ep['observations'][start:end]
                a_seq = ep['actions'][start:end]
                return torch.from_numpy(o_seq).float(), torch.from_numpy(a_seq).float()
            cum += max(0, L)
        raise IndexError
    
# Probabilistic
class BCPolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, num_layers=2):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_seq, hidden=None):
        x = self.net(obs_seq)
        mu = self.mean_head(x)  # (B, T, act_dim)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, hidden


from torch.distributions import Normal, Independent

def evaluate_bc(writer, model, env, batch_idx, batch_size, epoch, train=True):
    model.eval()
    returns = []
    rollouts = TRAIN_ROllOUTS if train else EVAL_ROLLOUTS
    for _ in range(rollouts):
        obs, _ = env.reset()
        done = False
        total_ret = 0.0
        while not done:
            obs_tensor = torch.from_numpy(obs[None, None]).float().to(DEVICE)
            with torch.no_grad():
                mu, std = model(obs_tensor)
            # Use deterministic mean at test time
            action = mu[0, 0].cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_ret += reward
        returns.append(total_ret)

    mean_ret = sum(returns) / len(returns)
    if train:
        step = (epoch - 1) * batch_size + batch_idx
        writer.add_scalar('train/mean_return', mean_ret, step)
        print(f" → {step} steps eval mean return: {mean_ret:.1f}")
    else:
        writer.add_scalar('eval/mean_return', mean_ret, epoch)
        print(f"Epoch {epoch:02d} eval mean return: {mean_ret:.1f}")
    model.train()


def train_bc(dataset, episodes, env):
    ds = SequenceDataset(episodes, seq_len=SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset observation space shape: {dataset.observation_space.shape[0]}")
    print(f"Env observation space shape: {env.observation_space.shape[0]}")
    model = BCPolicyMLP(dataset.observation_space.shape[0],
                        dataset.action_space.shape[0],
                        hidden_size=HIDDEN_SIZE,
                        num_layers=HIDDEN_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/bc_mlp_pusher/bc_mlp_pusher_{timestamp}")

    model.train()
    print("Training BC policy...")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        print(f"Training epoch {epoch} for {len(loader)} batches...")
        for batch_idx, (o_seq, a_seq) in enumerate(loader, start=1):
            o_seq, a_seq = o_seq.to(DEVICE), a_seq.to(DEVICE)

            opt.zero_grad()

            mu, std = model(o_seq)             # mu/std: (B, T, act_dim)
            # build a diagonal Gaussian over the action‐vector at each t
            dist = Independent(Normal(mu, std), 1) # event_dim=1 ⇒ log_prob sums over act_dim

            # log_prob: (B, T)
            log_prob = dist.log_prob(a_seq)       
            # average negative log‐likelihood over both batch and time:
            loss = -log_prob.mean()                

            loss.backward()
            opt.step()

            total_loss += loss.item() * o_seq.size(0)
            writer.add_scalar('train/batch_loss', loss.item(), (epoch - 1) * len(loader) + batch_idx)

            if batch_idx % 100 == 0:
                evaluate_bc(writer, model, env, batch_idx, len(loader), epoch, train=True)
        
        avg = total_loss / len(ds)
        writer.add_scalar('eval/epoch_loss', avg, epoch)
        print(f"Epoch {epoch:02d}  BC loss: {avg:.4f}")

        evaluate_bc(writer, model, env, len(loader), len(loader), epoch, train=False)


    writer.close()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model state_dict to {MODEL_PATH}")

    torch.save(opt.state_dict(), OPT_PATH)
    print(f"Saved optimizer state_dict to {OPT_PATH}")

def main():
    dataset = minari.load_dataset("mujoco/pusher/expert-v0")  
    print(f"Obs space: {dataset.observation_space}, Act space: {dataset.action_space}")
    print(f"Total episodes: {dataset.total_episodes}, Total steps: {dataset.total_steps}")

    env = gym.make("Pusher-v5")
    episodes = list(dataset.iterate_episodes())
    # Need to align episodes to account for terminal states
    aligned_episodes = []
    for ep in episodes:
        obs = ep.observations
        acts = ep.actions
        if obs.shape[0] == acts.shape[0] + 1:
            obs = obs[:-1]
        assert obs.shape[0] == acts.shape[0], (
            f"obs has {obs.shape[0]} vs acts {acts.shape[0]}"
        )

        aligned_episodes.append({'observations': obs, 'actions': acts})

    print(f"Loaded {len(aligned_episodes)} episodes; first lengths:",
        [ep['observations'].shape[0] for ep in aligned_episodes[:5]])
    
    train_bc(dataset, aligned_episodes, env)

if __name__ == "__main__":
    main()


