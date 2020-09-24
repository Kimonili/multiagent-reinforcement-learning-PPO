from collections import deque
from time import time
import psutil
from typing import List, Callable
import torch.nn as nn
import torch
from model.networks import ActorSilent, ActorSpeaker, Critic
import numpy as np
from utils.utils import compute_gae, init_env, calculate_score
import torch.multiprocessing
import os
import wandb
import random
from algorithm.memory import Trajectory

wandb.init(project="multiagent-particle-envs")


# my_dir = "runs/MPE/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


# print(my_dir)
# writer = SummaryWriter(log_dir=my_dir)

class PPOAgent:
    def __init__(self,
                 environment,
                 trajectory_size: int,
                 num_mini_batch: int,
                 gamma: float,
                 lam: float,
                 epsilon: float,
                 epochs: int,
                 entropy_weight: float,
                 hidden_size: int,
                 actor_lr: float,
                 critic_lr: float,
                 seed: int,
                 optimizer: Callable = torch.optim.Adam,
                 critic_weight=0.5,
                 max_grad_norm=5,
                 normalize_adv=False,
                 normalize_returns=False,
                 anneal_lr=False,
                 clip_epsilon_anneal=False,
                 clipped_value_loss=True,
                 torch_deterministic=True,
                 torch_benchmark=True,
                 partially_centralized=True,
                 fully_centralized=False,
                 multiprocessing=False,
                 ):

        self.env = environment
        self.trajectory_size = trajectory_size
        self.num_mini_batch = num_mini_batch
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epochs = epochs
        self.entropy_weight = entropy_weight
        self.hidden_size = hidden_size
        self.anneal_lr = anneal_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.critic_weight = critic_weight
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv
        self.normalize_returns = normalize_returns
        self.clipped_value_loss = clipped_value_loss
        self.clip_epsilon_anneal = clip_epsilon_anneal
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.torch_benchmark = torch_benchmark
        self.partially_centralized = partially_centralized
        self.fully_centralized = fully_centralized
        self.multiprocessing = multiprocessing

        self.critics = []
        self.actors = []
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.advantages = torch.zeros(len(self.env.agents), self.trajectory_size).to(self.device)
        self.returns = torch.zeros(len(self.env.agents), self.trajectory_size).to(self.device)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic
        torch.backends.cudnn.benchmark = self.torch_benchmark
        print(self.device)

        # initialize actor and critic networks, optimizers, observations, actions arrays
        # for each agent
        obs_size = [self.env.observation_space[agent].shape[0] for agent in range(len(self.env.agents))]
        self.centralized_states_serialize = np.zeros((self.trajectory_size, sum(obs_size)))
        self.mini_batch_size = self.trajectory_size // self.num_mini_batch

        # actors are always decentralized
        for i, agent in enumerate(self.env.agents):
            if self.env.agents[i].silent:
                self.action_size = [self.env.action_space[i].n for agent in range(len(self.env.agents))]
                torch.manual_seed(self.seed)
                actor = ActorSilent(obs_size[i], self.action_size[i], self.hidden_size).to(self.device)
                self.actors.append(actor)
            else:
                torch.manual_seed(self.seed)
                actor = ActorSpeaker(obs_size[i], self.hidden_size).to(self.device)
                self.actors.append(actor)

        if self.partially_centralized:  # if centralized critic networks - one for every agent
            for _ in range(len(self.env.agents)):
                torch.manual_seed(self.seed)
                critic = Critic(sum(obs_size), self.hidden_size).to(self.device)
                self.critics.append(critic)
        elif self.fully_centralized:  # if centralized critic networks - one for every team of agents
            torch.manual_seed(self.seed)
            critic_adv = Critic(sum(obs_size), self.hidden_size).to(self.device)
            critic_agent = Critic(sum(obs_size), self.hidden_size).to(self.device)
        else:  # if decentralized critic networks
            for agent in range(len(self.env.agents)):
                torch.manual_seed(self.seed)
                critic = Critic(obs_size[agent], self.hidden_size).to(self.device)
                self.critics.append(critic)

        # Log the network weights

        self.replay_buffer = [Trajectory(self.env, agent, self.device, self.trajectory_size).return_buffer()
                              for agent in range(len(self.env.agents))]
        self.actor_optimizers = [optimizer(actor.parameters(), lr=self.actor_lr)
                                 for actor in self.actors]
        self.critic_optimizers = [optimizer(critic.parameters(), lr=self.critic_lr)
                                  for critic in self.critics]

        # if instructed to have annealing learning rate then
        if self.anneal_lr:
            self.actor_lr_f = lambda f: f * self.actor_lr
            self.critic_lr_f = lambda f: f * self.critic_lr

        # if instructed to have annealing clip range
        if self.clip_epsilon_anneal:
            self.epsilon_anneal_f = lambda f: f * self.epsilon

    def select_action_serial(self, total_next_states):
        action_list = []
        for agent in range(len(self.env.agents)):
            # pass all the agents' states to every critic network as a result of centralized learning
            value = self.critics[agent].get_value(total_next_states)
            # pass only the agent's i state to the agent's i actor network as a result of decentralized execution
            action, log_probs , _ = self.actors[agent].get_action(self.replay_buffer[agent]['states'][self.step])


            if self.env.agents[agent].silent:
                # 1 for the action taken, 0 otherwise
                action_vector = np.zeros(self.action_size[agent])
                action_vector[action] = 1
                action_list.append(action_vector)
                self.replay_buffer[agent]['actions'][self.step] = action
                self.replay_buffer[agent]['log_probs'][self.step] = log_probs
                self.replay_buffer[agent]['values'][self.step] = value
            else:
                action_vector_move = np.zeros(5)
                action_vector_comm = np.zeros(2)
                action_vector_move[action[0]] = 1
                action_vector_comm[action[1]] = 1
                action_list.append([action_vector_move, action_vector_comm])
                # save the action, value, state taken in the replay buffer
                self.replay_buffer[agent]['actions'][self.step] = torch.tensor(action)
                self.replay_buffer[agent]['log_probs'][self.step] = torch.tensor(log_probs)
                self.replay_buffer[agent]['values'][self.step] = value


        return action_list

    def make_action(self, action_vector):

        next_state, reward, done, _ = self.env.step(action_vector)
        for agent in range(len(self.env.agents)):
            done[agent] = torch.tensor(int(done[agent]), dtype=torch.float32).to(self.device)
            self.replay_buffer[agent]['rewards'][self.step] = reward[agent]

        return next_state, done

    def update(self,
               replay_buffer,
               advantages,
               returns,
               actors,
               critics,
               centralized_states_serialize):

        actors_loss = []
        critics_loss = []
        # ids = np.arange(self.trajectory_size)
        for agent, _ in enumerate(self.env.agents):
            ids = np.arange(self.trajectory_size)
            for epoch in range(self.epochs):
                np.random.seed(self.seed)
                np.random.shuffle(ids)
                for start in range(0, self.trajectory_size, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    minibatch_ind = ids[start:end]

                    if self.env.agents[agent].silent:
                        _, newlogproba, entropy = actors[agent].get_action(replay_buffer[agent]['states'][minibatch_ind],
                                                                           replay_buffer[agent]['actions'][minibatch_ind])
                        ratio = (newlogproba - replay_buffer[agent]['log_probs'][minibatch_ind]).exp()

                    else:
                        _, newlogproba, entropy = actors[agent].get_action(replay_buffer[agent]['states'][minibatch_ind],
                                                                           torch.tensor(list(zip(*replay_buffer[agent]['actions']))[0])[minibatch_ind],
                                                                           torch.tensor(list(zip(*replay_buffer[agent]['actions']))[1])[minibatch_ind])
                        newlogproba = torch.tensor(list(zip(newlogproba[0], newlogproba[1])), requires_grad=True)
                        entropy = torch.tensor(list(zip(entropy[0], entropy[1])), requires_grad=True)
                        ratio = (newlogproba - replay_buffer[agent]['log_probs'][minibatch_ind]).exp()
                        ratio = torch.tensor([torch.mean(ratio[i]) for i in range(len(ratio))], requires_grad=True)
                    if self.normalize_adv:
                        advantages_minib = (advantages[agent][minibatch_ind] - advantages[agent][
                            minibatch_ind].mean()) / (
                                                   advantages[agent][minibatch_ind].std() + 1e-8)
                        advantages_minib = advantages_minib.detach()
                    else:
                        advantages_minib = advantages[agent][minibatch_ind].detach()

                    # ratio = (newlogproba - replay_buffer[agent]['log_probs'][minibatch_ind]).exp()
                    # if self.env.agents[agent].silent:
                    #     surr_loss = -advantages_minib * ratio
                    # else:
                    #     ratio = torch.tensor([torch.mean(ratio[i]) for i in range(len(ratio))])
                    #    surr_loss = -advantages_minib * ratio

                    # actor loss
                    surr_loss = -advantages_minib * ratio
                    clipped_surr_loss = -advantages_minib * torch.clamp(ratio, 1 - self.epsilon,
                                                                        1 + self.epsilon)
                    actor_loss_max = torch.max(surr_loss, clipped_surr_loss).mean()
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss_max - self.entropy_weight * entropy_loss

                    # critic_loss
                    new_values = critics[agent].get_value(torch.tensor(centralized_states_serialize,
                                                                       dtype=torch.float32))[minibatch_ind]
                    returns_minib = returns[agent][minibatch_ind].detach()
                    if self.clipped_value_loss:
                        if self.normalize_returns:
                            returns_minib = (
                                    (returns_minib - returns_minib.mean()) / (returns_minib.std() + 1e-8)).detach()
                        critic_loss_unclipped = (new_values - returns_minib.detach()) ** 2
                        value_clipped = replay_buffer[agent]['values'][minibatch_ind] + torch.clamp(new_values -
                                                                                                    replay_buffer[
                                                                                                        agent]['values']
                                                                                                    [minibatch_ind],
                                                                                                    - self.epsilon,
                                                                                                    self.epsilon)
                        critic_loss_clipped = (value_clipped - returns_minib.detach()) ** 2
                        critic_loss_max = torch.max(critic_loss_clipped, critic_loss_unclipped)
                        critic_loss = 0.5 * critic_loss_max.mean() * self.critic_weight
                    else:
                        critic_loss = 0.5 * (
                                new_values - returns_minib.detach() ** 2).mean() * self.critic_weight

                    loss = actor_loss + critic_loss
                    # critic backward pass
                    self.critic_optimizers[agent].zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1.0)
                    self.critic_optimizers[agent].step()
                    # actor backward pass
                    self.actor_optimizers[agent].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 2.3)
                    self.actor_optimizers[agent].step()
            actors_loss.append(actor_loss)
            critics_loss.append(critic_loss)

        return actors_loss, critics_loss

    def train(self, num_env_steps=1000000):

        threshold = 20
        de1 = deque(maxlen=threshold)
        score_avg = np.full(shape=(len(self.env.agents), 0), fill_value=de1)
        de2 = deque(maxlen=self.trajectory_size)
        total_score_per_update = np.full(shape=(len(self.env.agents), 0), fill_value=de2)
        actor_losses, critic_losses = [], []
        score = np.empty(len(self.env.agents))
        num_updates = int(num_env_steps) // self.trajectory_size
        num_steps = self.trajectory_size
        next_done = torch.zeros(len(self.env.agents))
        # self.env.seed(self.seed)
        if self.multiprocessing:
            import torch.multiprocessing as mp
            select_pool = mp.Pool(os.cpu_count())
        # actor_total_norms = []
        # critic_total_norms = []
        for update in range(1, num_updates):
            if self.multiprocessing:
                if psutil.virtual_memory().percent > 90:
                    select_pool.close()
                    select_pool.join()
                    select_pool = mp.Pool(os.cpu_count())
            np.random.seed(self.seed)
            next_state = self.env.reset()
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                actor_lr_now = self.actor_lr_f(frac)
                critic_lr_now = self.critic_lr_f(frac)
                for i, agent in enumerate(self.env.agents):
                    self.actor_optimizers[i].param_groups[0]['lr'] = actor_lr_now
                    self.critic_optimizers[i].param_groups[0]['lr'] = critic_lr_now
            if self.clip_epsilon_anneal:
                frac = 1.0 - (update - 1.0) / num_updates
                epsilon_now = self.epsilon_anneal_f(frac)
                self.epsilon = epsilon_now
            else:
                print("Constant clip epsilon")

            start = time()
            for self.step in range(0, num_steps):
                # given a state agent selects an action
                with torch.no_grad():
                    for agent in range(len(self.env.agents)):
                        next_state[agent] = torch.tensor(next_state[agent], dtype=torch.float32).to(self.device)
                        self.replay_buffer[agent]['states'][self.step] = next_state[agent]
                        self.replay_buffer[agent]['masks'][self.step] = next_done[agent]
                    # fix arguments to be valid for multiprocessing
                    self.step_iter = np.full(shape=len(self.env.agents), fill_value=self.step, dtype=np.int)
                    self.centralized_states_serialize[self.step] = np.concatenate(next_state, axis=0)
                    total_next_states = torch.cat(next_state, dim=0)
                    centralized_states_multiprocess = [total_next_states] * len(self.env.agents)
                    # select action for every agent
                    if self.multiprocessing:
                        action_vector = select_pool.starmap(Trajectory.select_action_parallel, zip(self.replay_buffer,
                                                                                                   self.actors,
                                                                                                   self.critics,
                                                                                                   self.step_iter,
                                                                                                   self.action_size,
                                                                                                   centralized_states_multiprocess))
                    else:
                        action_vector = self.select_action_serial(total_next_states)
                    # make that action and take a step to the environment
                    next_state, next_done = self.make_action(action_vector)
                # env.render()
            end = time() - start

            for i in range(len(self.env.agents)):
                print("Agent {} scored {} points on this update".
                      format(i, torch.mean(self.replay_buffer[i]['rewards'])))

            print(end)
            last_value = [
                self.critics[agent].get_value(torch.tensor(np.concatenate(next_state, axis=0), dtype=torch.float32))
                    .to(self.device) for agent in range(len(self.env.agents))]

            compute_gae(last_value, self.replay_buffer, self.gamma,
                        self.lam, self.device, self.advantages,
                        self.returns, self.env)
            actors_loss, critics_loss = self.update(self.replay_buffer,
                                                    self.advantages,
                                                    self.returns,
                                                    self.actors,
                                                    self.critics,
                                                    self.centralized_states_serialize)

            # actor_norms = []
            # for actor in range(len(self.actors)):
            #     total_norm = 0
            #     for p in self.actors[actor].parameters():
            #         param_norm = p.grad.data.norm()
            #         total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** (1. / 2)
            #     actor_norms.append(total_norm)
            # actor_total_norms.extend([actor_norms])
            # critic_norms = []
            # for critic in range(len(self.critics)):
            #     total_norm = 0
            #     for p in self.critics[critic].parameters():
            #         param_norm = p.grad.data.norm()
            #         total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** (1. / 2)
            #     critic_norms.append(total_norm)
            # critic_total_norms.extend([critic_norms])
            # if update == 10:
            #     print('stop')

            for agent, _ in enumerate(self.env.agents):
                for i in range(0, self.trajectory_size):
                    if agent < 3:
                        wandb.log({'update': update,
                                   f'reward_adversary_{agent}_per_update': torch.mean(
                                       self.replay_buffer[agent]['rewards']),
                                   f'reward_adversary_{agent}_per_step': self.replay_buffer[agent]['rewards'][i]})

                    else:
                        wandb.log({'update': update,
                                   f'reward_agent_{agent}_per_update': torch.mean(self.replay_buffer[agent]['rewards']),
                                   f'reward_agent_{agent}_per_step': self.replay_buffer[agent]['rewards'][i]})
            de1.append(torch.mean(self.replay_buffer[3]['rewards']))
        if self.multiprocessing:
            select_pool.close()
            select_pool.join()


env, seed = init_env(gym_id='predator_pray_bounded_comm', seed=16)

agent = PPOAgent(environment=env, trajectory_size=612, num_mini_batch=8, gamma=0.999, lam=0.9, epsilon=0.2,
                 epochs=10, entropy_weight=0.005, hidden_size=64,
                 actor_lr=1e-5, critic_lr=5e-5, seed=seed)

if __name__ == '__main__':
    agent.train()
