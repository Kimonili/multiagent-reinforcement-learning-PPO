import torch
import numpy as np
from multiagent_particle_envs.make_env import make_env
from collections import deque


# https://github.com/MrSyee/pg-is-all-you-need/blob/master/02.PPO.ipynb
# function for the calculation of the general advantage estimation


def compute_gae(last_value,
                replay_buffer,
                gamma,
                lam,
                device,
                advantages,
                returns,
                env):
    gae = 0
    for agent in range(len(env.agents)):
        for step in reversed(range(len(replay_buffer[agent]['rewards']))):
            if step == len(replay_buffer[agent]['rewards'])-1:
                next_done = torch.tensor([1.0]).to(device)
                next_non_terminal = 1.0 - next_done
                next_value = last_value[agent].to(device)
            else:
                next_non_terminal = 1.0 - replay_buffer[agent]['masks'][step + 1]
                next_value = replay_buffer[agent]['values'][step + 1]
        # delta = torch.empty(len(agents))
            delta = replay_buffer[agent]['rewards'][step] + gamma * next_value * next_non_terminal \
                    - replay_buffer[agent]['values'][step]
            advantages[agent][step] = gae = delta + gamma * lam * next_non_terminal * gae
        returns[agent] = advantages[agent] + replay_buffer[agent]['values']
    # return returns, advantages
    # for step in reversed(range(len(rewards[idx]))):
    #     if step == len(rewards[idx]) - 1:
    #         next_done = torch.tensor([1.0]).to(cuda_or_cpu)
    #         next_non_terminal = 1.0 - next_done
    #         next_values = last_value[idx].to(cuda_or_cpu)
    #     else:
    #         next_non_terminal = 1.0 - masks[idx][step + 1]
    #         next_values = values[idx][step + 1]
    #
    #     delta = rewards[idx][step] + gamma * next_values * next_non_terminal - values[idx][step]
    #     advantages[idx][step] = gae = delta + gamma * lam * next_non_terminal * gae
    #     # we append the returns from left to right as we had previously reversed the list
    #     # of steps (we were looping from the last step to the fist)
    #     # in this way we return back to the initial order
    # returns[idx] = advantages[idx] + values[idx]

    # return returns[idx], advantages[idx]


def calculate_score(agents, trajectory_size, step, rewards_list):
    score_threshold = 100
    de1 = deque(maxlen=score_threshold)
    score_avg = np.full(shape=(len(agents), 0), fill_value=de1)
    de2 = deque(maxlen=trajectory_size)
    total_score_per_update = np.full(shape=(len(agents), 0), fill_value=de2)
    actor_losses, critic_losses = [], []
    score = np.empty(len(agents))
    for i in range(len(agents)):
        if step == 0:
            score[i] = rewards_list[i][step]
        else:
            score[i] += rewards_list[i][step]
        total_score_per_update[i] = score[i]
        score_avg[i] = score[i]
    return total_score_per_update, score_avg


def init_env(gym_id: str, seed: int):
    env = make_env(gym_id, discrete_action=True)
    env.seed(seed)
    return env, seed