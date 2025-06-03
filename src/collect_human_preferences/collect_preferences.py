import gymnasium as gym
import torch
import imageio
from torch.distributions import Normal, Independent
import numpy as np
from ..train_humanoid_baseline import BCPolicyRNN
from ..train_pusher_baseline import BCPolicyMLP
import pygame
import sys
import pickle
import os
import random

from .utils import MODEL_REGISTRY, load_env, load_models, rollout_trajectory, display_videos



# TODO: make sure this is correct
def main(args):
    if not args.save_path.endswith(".pkl"):
        raise ValueError("save-path must end in .pkl for pickle serialization.")

    env = load_env(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_1, model_2 = load_models(obs_dim, act_dim,
                                   args.model_1_path, args.model_2_path,
                                   args.model_class, args.model_class,
                                   device=args.device)

    # if we're resuming, load in the past preferences first
    if os.path.exists(args.save_path) and args.resume:
        with open(args.save_path, "rb") as f:
            preferences = pickle.load(f)
    else:
        preferences = []

    for i in range(args.num_pairs):

        print(f"Collecting pair {i + 1}/{args.num_pairs}...")

        # Use same seed for both trajectories in the pair
        traj_1 = rollout_trajectory(model_1, env, args.max_steps, args.device, seed=len(preferences))
        traj_2 = rollout_trajectory(model_2, env, args.max_steps, args.device, seed=len(preferences))

        # Randomize which is shown on the left/right
        if random.random() < 0.5:
            left_traj, right_traj = traj_1, traj_2
            left_label, right_label = args.model_1_label, args.model_2_label
            left_is_model1 = True
        else:
            left_traj, right_traj = traj_2, traj_1
            left_label, right_label = args.model_2_label, args.model_1_label
            left_is_model1 = False

        choice = display_videos(left_traj["frames"], right_traj["frames"],
                              label_left=left_label, label_right=right_label,
                              last_frame_only=False, fps=60)

        # Determine which trajectory is model_1
        chosen_traj = left_traj if choice == 1 else right_traj
        rejected_traj = right_traj if choice == 1 else left_traj
        chosen_is_model1 = (choice == 1 and left_is_model1) or (choice == 2 and not left_is_model1)

        preference = {
            "chosen_obs": chosen_traj["observations"],
            "chosen_act": chosen_traj["actions"],
            "rejected_obs": rejected_traj["observations"],
            "rejected_act": rejected_traj["actions"],
            "metadata": {
                "pair_index": len(preferences),
                "env": args.env_name,
                "left_model": left_label,
                "right_model": right_label,
                "preferred_side": "left" if choice == 1 else "right",
                "preferred_model": left_label if choice == 1 else right_label,
                "chosen_is_model1": chosen_is_model1
            }
        }
        preferences.append(preference)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(preferences, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-1-path", type=str, required=True)
    parser.add_argument("--model-2-path", type=str, required=True)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--model-class", type=str, required=True)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-1-label", type=str, default="Model 1")
    parser.add_argument("--model-2-label", type=str, default="Model 2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Resume appending to an existing preferences file if it exists.")

    args = parser.parse_args()

    if args.model_class not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class: {args.model_class}. Must be one of: {list(MODEL_REGISTRY.keys())}")
    args.model_class = MODEL_REGISTRY[args.model_class]
    main(args)


"""
Example usage in terminal 

python -m src.collect_human_preferences.collect_preferences \
  --model-1-path policies/bcc_rnn_humanoid_deterministic/bc_rnn_humanoid.pth \
  --model-2-path policies/bcc_rnn_humanoid_deterministic/bc_rnn_humanoid.pth \
  --env-name Humanoid-v5 \
  --model-class BCPolicyRNN \
  --num-pairs 5 \
  --max-steps 750 \
  --save-path src/collect_human_preferences/preferences/humanoid_bc_vs_dpo.pkl \
  --resume

  python -m src.collect_human_preferences.collect_preferences \
  --model-1-path policies/bc_mlp_pusher/bc_mlp_pusher.pth \
  --model-2-path policies/bc_mlp_pusher/bc_mlp_pusher.pth \
  --env-name Pusher-v5 \
  --model-class BCPolicyMLP \
  --num-pairs 5 \
  --max-steps 750 \
  --save-path src/collect_human_preferences/preferences/pusher_same_start.pkl \
  --resume
"""