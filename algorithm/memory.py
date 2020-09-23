import torch
import numpy as np
import gym


class Trajectory:
    def __init__(self,
                 env: gym.Env,
                 agent,
                 device,
                 trajectory_size
                 ):
        self.env = env
        self.obs_size = self.env.observation_space[agent].shape[0]
        self.action_size = self.env.action_space[agent].n
        self.states = torch.zeros(trajectory_size, self.obs_size).to(device)
        self.actions = torch.zeros(trajectory_size).to(device)
        self.rewards = torch.ones(trajectory_size).to(device)
        self.values = torch.ones(trajectory_size).to(device)
        self.masks = torch.zeros(trajectory_size).to(device)
        self.log_probs = torch.zeros(trajectory_size).to(device)

    def return_buffer(self):
        values = [self.states, self.actions, self.rewards, self.values, self.masks, self.log_probs]
        keys = ['states', 'actions', 'rewards', 'values', 'masks', 'log_probs']

        return dict(zip(keys, values))

    @staticmethod
    def select_action_parallel(replay_buffer,
                               actors,
                               critics,
                               step,
                               action_space,
                               centralized_states_multiprocess):
        # pass all the agents' states to every critic network as a result of centralized learning
        value = critics.get_value(centralized_states_multiprocess)
        # pass only the agent's i state to the agent's i actor network as a result of decentralized execution
        action, replay_buffer['log_probs'][step], _ = actors.get_action(replay_buffer['states'][step])
        # 1 for the action taken, 0 otherwise
        action_vector = np.zeros(action_space)
        action_vector[action] = 1
        # save the action, value, state taken in the replay buffer
        replay_buffer['actions'][step] = action
        replay_buffer['values'][step] = value

        return action_vector
