from gymnasium.wrappers import RecordEpisodeStatistics
import gymnasium as gym

def generate_preference_dataset(task_name: str, num_pairs: int):
    env = RecordEpisodeStatistics(gym.make(task_name))

    preference_data = []

    for _ in range(num_pairs):
        obs_0, _ = env.reset()
        trajs = []

        for _ in range(2):  # generate 2 candidate trajectories per initial state (prompt)
            obs, actions, rewards, dones = [], [], [], []
            done, trunc = False, False
            obs_t = obs_0.copy()
            while not (done or trunc):
                action = policy(obs_t) # this is the pretrained base policy
                obs_t, reward, done, trunc, _ = env.step(action)
                obs.append(obs_t)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

            trajs.append({
                "observations": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones
            })

        # simple auto-label
        r0, r1 = sum(trajs[0]["rewards"]), sum(trajs[1]["rewards"])
        preference = 0 if r0 > r1 else 1

        preference_data.append({
            "prompt": obs_0.tolist(),
            "trajectories": trajs,
            "preferred": preference
        })

    return preference_data
