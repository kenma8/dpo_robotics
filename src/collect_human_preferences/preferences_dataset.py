import torch
from torch.utils.data import Dataset
import pickle


class PreferencesDataset(Dataset):
    def __init__(self, *paths_to_pref_pkls):
        self.data = []
        for path in paths_to_pref_pkls:
            with open(path, "rb") as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_obs = torch.tensor(item["chosen_obs"], dtype=torch.float32)
        chosen_act = torch.tensor(item["chosen_act"], dtype=torch.float32)
        rejected_obs = torch.tensor(item["rejected_obs"], dtype=torch.float32)
        rejected_act = torch.tensor(item["rejected_act"], dtype=torch.float32)

        return {
            "chosen_obs": chosen_obs,  # (T, obs_dim)
            "chosen_act": chosen_act,  # (T, act_dim)
            "rejected_obs": rejected_obs,  # (T, obs_dim)
            "rejected_act": rejected_act  # (T, act_dim)
        }


import pickle

with open("preferences/pusher_same_start.pkl", "rb") as f:
    preferences = pickle.load(f)

print(f"Number of preference examples: {len(preferences)}")
