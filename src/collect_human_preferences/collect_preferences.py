import gymnasium as gym
import torch
import imageio
from torch.distributions import Normal, Independent
import numpy as np
from ..train_humanoid_baseline import BCPolicyRNN
import pygame
import sys
import pickle
import os
import random


# TODO: add more models as we get more models. rn, we can only use BCPolicyRNN
MODEL_REGISTRY = {
    "BCPolicyRNN": BCPolicyRNN,
    # Add more models as needed
}


# TODO: revisit when we have other models. Idk if this will work for other model types
def rollout_trajectory(model, env, max_steps=1000, device="cpu"):
    obs, _ = env.reset()
    frames = []
    observations = []
    actions = []
    rewards = []
    hidden = None

    for _ in range(max_steps):
        observations.append(obs)

        obs_tensor = torch.from_numpy(obs[None, None]).float().to(device)
        with torch.no_grad():
            # get model output
            mu, std, hidden = model(obs_tensor, hidden)
            # divide std by 3 to get a narrower/taller normal distr.
            # (make model take actions closer to mean)
            dist = Independent(Normal(mu, std / 3), 1)
            # sample action and clip it to stay in the bounds of the env action space
            action = dist.sample()[0, 0].cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        rewards.append(reward)
        if terminated or truncated:
            break

    return {
        "frames": frames,
        "observations": np.stack(observations),  # (T, obs_dim)
        "actions": np.stack(actions),            # (T, act_dim)
        "rewards": np.array(rewards)             # (T,)
    }


# TODO come back and make this better for loading in the model and such
def load_model(model_class, model_path, obs_dim, act_dim, device="cpu"):
    model = model_class(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def load_models(obs_dim, act_dim,
                model_1_path, model_2_path,
                model_1_class, model_2_class=None,
                device="cpu"):
    """
    :param obs_dim: obs dimension for models
    :param act_dim: action dimensions for models
    :param model_1_path: path to .pth file holding model_1's weights
    :param model_2_path: path to .pth file holding model_1's weights
    :param model_1_class: class type of model_1
    :param model_2_class: class type of model_2 (defaults to model_1's class type)
    :param device: cpu, gpu, etc
    :return: tuple of pytorch models: (model_1, model_2)
    """
    model_1 = load_model(model_1_class,
                         model_1_path,
                         obs_dim, act_dim,
                         device=device)
    model_2 = load_model(model_2_class,
                         model_2_path if model_2_class is not None else model_1_class,
                         obs_dim,
                         act_dim,
                         device=device)
    return model_1, model_2


def load_gym_env(env_name):
    """
    :param env_name: gym environment name as a string
    :return: gym environment
    """
    return gym.make(env_name, render_mode="rgb_array")


# TODO: fill in the portion for handling non-gym environments
def load_env(env_name):
    # if the environment name is a gym environment, load gym env
    gym_environment_names = set(gym.envs.registry.keys())
    if env_name in gym_environment_names:
        return load_gym_env(env_name)
    # we currently only support gym environments, but change this when we use the other
    else:
        raise ValueError(f"{env_name!r} is not a valid Gym environment.")


def display_videos(frames_left, frames_right, label_left="Model 1", label_right="Model 2", fps=30):
    pygame.init()

    # Assume both videos are same shape
    frame_height, frame_width, _ = frames_left[0].shape
    padding = 10
    total_width = 2 * frame_width + padding
    screen = pygame.display.set_mode((total_width, frame_height))
    pygame.display.set_caption("Preference Comparison")

    font = pygame.font.SysFont(None, 30)
    label_left_surface = font.render(label_left, True, (255, 255, 255))
    label_right_surface = font.render(label_right, True, (255, 255, 255))

    clock = pygame.time.Clock()
    # TODO: look at this for allowing one to fall and the other keep going
    # TODO: currently, this stops both when one falls
    num_frames = min(len(frames_left), len(frames_right))

    for i in range(num_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        frame_left = pygame.surfarray.make_surface(frames_left[i].swapaxes(0, 1))
        frame_right = pygame.surfarray.make_surface(frames_right[i].swapaxes(0, 1))

        screen.blit(frame_left, (0, 0))
        screen.blit(frame_right, (frame_width + padding, 0))

        screen.blit(label_left_surface, (10, 10))
        screen.blit(label_right_surface, (frame_width + padding + 10, 10))

        pygame.display.flip()
        clock.tick(fps)

    # Wait for user input
    print("Press 0 (left) or 1 (right) to indicate which trajectory you prefer.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    pygame.quit()
                    return 0
                elif event.key == pygame.K_1:
                    pygame.quit()
                    return 1


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

    for i in range(args.num_pairs):
        traj_1 = rollout_trajectory(model_1, env, args.max_steps, args.device)
        traj_2 = rollout_trajectory(model_2, env, args.max_steps, args.device)

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
                                label_left=left_label, label_right=right_label)

        # Determine which trajectory is model_1
        chosen_traj = left_traj if choice == 0 else right_traj
        rejected_traj = right_traj if choice == 0 else left_traj
        chosen_is_model1 = (choice == 0 and left_is_model1) or (choice == 1 and not left_is_model1)

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
                "preferred_side": "left" if choice == 0 else "right",
                "preferred_model": left_label if choice == 0 else right_label,
                "chosen_is_model1": chosen_is_model1
            }
        }
        preferences.append(preference)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(preferences, f)
        print(preferences)


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
  --model-1-path bc_rnn_humanoid.pth \
  --model-2-path bc_rnn_humanoid.pth \
  --env-name Humanoid-v5 \
  --model-class BCPolicyRNN \
  --num-pairs 5 \
  --max-steps 750 \
  --save-path src/collect_human_preferences/preferences/humanoid_bc_vs_dpo.pkl \
  --resume
"""