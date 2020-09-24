import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils.mish import Mish
import numpy as np

mish_activation = Mish()
tanh_activation = nn.Tanh()
relu_activation = nn.ReLU()


class ActorSilent(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size,
                 activation=tanh_activation):
        super(ActorSilent, self).__init__()

        self.action = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            activation,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation,
            layer_init(nn.Linear(hidden_size, action_size), std=0.01),
        )

    def forward(self):
        raise NotImplementedError

    def get_action(self, state, action=None):
        logits = self.action(state)
        probs = Categorical(logits=logits)
        # print('The logits passed to Categorical are: ', logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class ActorSpeaker(nn.Module):
    def __init__(self,
                 obs_size,
                 hidden_size,
                 activation=tanh_activation):
        super(ActorSpeaker, self).__init__()

        self.feature_layers = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            activation,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation
        )

        self.action_move = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 4), std=0.01)
        )

        self.action_comm = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 2), std=0.01)
        )

    def forward(self):
        raise NotImplementedError

    def get_action(self, state, action_move=None, action_comm=None):
        feature = self.feature_layers(state)
        logits_move = self.action_move(feature)
        logits_comm = self.action_comm(feature)
        probs_move = Categorical(logits=logits_move)
        probs_comm = Categorical(logits=logits_comm)
        if action_move is None and action_comm is None:
            action_move = probs_move.sample()
            action_comm = probs_comm.sample()
        return (action_move, action_comm),\
               (probs_move.log_prob(action_move), probs_comm.log_prob(action_comm)), \
               (probs_move.entropy(), probs_comm.entropy())


class Critic(nn.Module):
    def __init__(self, obs_size: int, hidden_size, activation=tanh_activation):
        """Initialize."""
        super(Critic, self).__init__()

        self.value = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            activation,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation,
            layer_init(nn.Linear(hidden_size, 1), std=1.)
        )

    def forward(self):
        raise NotImplementedError

    def get_value(self, state):
        return self.value(state)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
