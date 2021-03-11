import numpy as np
import scipy.signal

import torch
import torch.nn as nn



def mlp(sizes, activation, output_activation=nn.Identity, layernorm=True):
    layers = []
    for j in range(len(sizes)-1):
        if (j < len(sizes)-2) and layernorm:
            layers += [
                nn.Linear(sizes[j], sizes[j+1]), 
                nn.LayerNorm(sizes[j+1], elementwise_affine=False),
                activation()]
        elif j < len(sizes)-2:
            layers += [
                nn.Linear(sizes[j], sizes[j+1]), 
                activation()] 
        else:
            layers += [
                nn.Linear(sizes[j], sizes[j+1]), 
                output_activation()] 
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, layernorm):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh, layernorm=layernorm)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        if len(obs.shape) == 1:
            return self.act_limit * self.pi(obs.unsqueeze(0)).squeeze()
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, layernorm):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, layernorm=layernorm)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, layernorm=True):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, layernorm=layernorm)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, layernorm=layernorm)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, layernorm=layernorm)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
