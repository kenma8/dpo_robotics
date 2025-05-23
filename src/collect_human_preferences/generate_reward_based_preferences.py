import pickle
import numpy as np
import os
from multiprocessing import Pool
from functools import partial

from .utils import load_env, load_models, rollout_trajectory, MODEL_REGISTRY, display_videos

def generate_trajectory_pair(models_and_env, max_steps, device):
    model_1, model_2, env = models_and_env
    
    traj_1 = rollout_trajectory(model_1, env, max_steps, device)
    traj_2 = rollout_trajectory(model_2, env, max_steps, device)
    
    total_reward_1 = np.sum(traj_1["rewards"])
    total_reward_2 = np.sum(traj_2["rewards"])
    
    if total_reward_1 >= total_reward_2:
        chosen_traj = traj_1
        rejected_traj = traj_2
        chosen_str = "run_1"
    else:
        chosen_traj = traj_2
        rejected_traj = traj_1
        chosen_str = "run_2"
        
    return {
        "chosen_obs": chosen_traj["observations"],
        "chosen_act": chosen_traj["actions"],
        "rejected_obs": rejected_traj["observations"],
        "rejected_act": rejected_traj["actions"],
        "metadata": {
            "reward_chosen": np.sum(chosen_traj["rewards"]),
            "reward_rejected": np.sum(rejected_traj["rewards"]),
            "chosen_str": chosen_str
        }
    }

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

    if os.path.exists(args.save_path) and args.resume:
        with open(args.save_path, "rb") as f:
            preferences = pickle.load(f)
    else:
        preferences = []

    if args.parallel:
        # Parallel generation
        num_workers = min(os.cpu_count(), args.num_pairs)
        envs = [load_env(args.env_name) for _ in range(num_workers)]
        models_and_envs = [(model_1, model_2, env) for env in envs]
        
        generate_pair = partial(generate_trajectory_pair, 
                              max_steps=args.max_steps,
                              device=args.device)

        with Pool(num_workers) as pool:
            new_preferences = pool.map(generate_pair, models_and_envs)

        # Add metadata that requires the full preferences list
        start_idx = len(preferences)
        for i, pref in enumerate(new_preferences):
            pref["metadata"]["pair_index"] = start_idx + i
            pref["metadata"]["env"] = args.env_name
            preferences.append(pref)
            print(f"Pair {start_idx + i + 1}: Reward comparison {pref['metadata']['reward_chosen']:.2f} vs {pref['metadata']['reward_rejected']:.2f}")
    else:
        # Sequential generation
        for i in range(args.num_pairs):
            pref = generate_trajectory_pair((model_1, model_2, env), args.max_steps, args.device)
            pref["metadata"]["pair_index"] = len(preferences)
            pref["metadata"]["env"] = args.env_name
            preferences.append(pref)
            print(f"Pair {pref['metadata']['pair_index']}: Chose {pref['metadata']['chosen_str']} Reward comparison {pref['metadata']['reward_chosen']:.2f} vs {pref['metadata']['reward_rejected']:.2f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(preferences, f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--model-1-path", type=str, required=True)
    parser.add_argument("--model-2-path", type=str, required=True)
    parser.add_argument("--model-class", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel trajectory generation")

    args = parser.parse_args()

    if args.model_class not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class: {args.model_class}. Must be one of: {list(MODEL_REGISTRY.keys())}")
    args.model_class = MODEL_REGISTRY[args.model_class]
    main(args)
