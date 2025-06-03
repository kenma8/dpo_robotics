import gymnasium as gym
import torch
from torch.distributions import Normal, Independent
import numpy as np
from ..train_humanoid_baseline import BCPolicyRNN
from ..train_pusher_baseline import BCPolicyMLP
import pygame
import sys


# TODO: add more models as we get more models. rn, we can only use BCPolicyRNN
MODEL_REGISTRY = {
    "BCPolicyRNN": BCPolicyRNN,
    "BCPolicyMLP": BCPolicyMLP,
    # Add more models as needed
}


# TODO: revisit when we have other models. Idk if this will work for other model types
def rollout_trajectory(model, env, max_steps=1000, device="cpu", seed=None):
    """
    Roll out a trajectory with optional seed for reproducible initial state
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)  # Set seed for reproducible initial state
    else:
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
            # TODO: change this for other models / for different experiments
            std = torch.clamp(std * 1.25, min=1e-4)
            dist = Independent(Normal(mu, std), 1)
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


def display_videos(frames_left, frames_right, label_left="Model 1", label_right="Model 2", fps=30, last_frame_only=False):
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

    if last_frame_only:
        # Get last frame from shorter trajectory
        min_length = min(len(frames_left), len(frames_right))
        last_frame_idx = min_length - 1
        last_frame_left = frames_left[last_frame_idx]
        last_frame_right = frames_right[last_frame_idx]

        # Display last frames
        frame_left = pygame.surfarray.make_surface(last_frame_left.swapaxes(0, 1))
        frame_right = pygame.surfarray.make_surface(last_frame_right.swapaxes(0, 1))
        print("Press 1 (left) or 2 (right) to indicate which trajectory you prefer.")
    else:
        clock = pygame.time.Clock()
        num_frames = min(len(frames_left), len(frames_right))
        frame_idx = 0
        done_playing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    pygame.quit()
                    return 1
                elif event.key == pygame.K_2:
                    pygame.quit()
                    return 2

        if last_frame_only:
            screen.blit(frame_left, (0, 0))
            screen.blit(frame_right, (frame_width + padding, 0))
            screen.blit(label_left_surface, (10, 10))
            screen.blit(label_right_surface, (frame_width + padding + 10, 10))
            pygame.display.flip()
        else:
            if frame_idx < num_frames:
                frame_left = pygame.surfarray.make_surface(frames_left[frame_idx].swapaxes(0, 1))
                frame_right = pygame.surfarray.make_surface(frames_right[frame_idx].swapaxes(0, 1))

                screen.blit(frame_left, (0, 0))
                screen.blit(frame_right, (frame_width + padding, 0))
                screen.blit(label_left_surface, (10, 10))
                screen.blit(label_right_surface, (frame_width + padding + 10, 10))

                pygame.display.flip()
                clock.tick(fps)
                frame_idx += 1
            elif not done_playing:
                print("Press 1 (left) or 2 (right) to indicate which trajectory you prefer.")
                done_playing = True