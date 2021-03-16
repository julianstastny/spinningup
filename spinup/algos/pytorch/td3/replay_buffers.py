from copy import deepcopy
import itertools
import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def calculate_nstep_returns(rewards, n, gamma):
    returns = np.array(rewards).astype(np.float64)
    reward = np.copy(returns).astype(np.float64)
    for _ in range(n - 1):
        reward = np.roll(reward, -1, 0)
        reward[-1] = 0.0
        reward *= gamma
        returns += reward
    return returns

class MultiStepIntermediateBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self._ready_to_reset = True

    def reset(self):
        assert self._ready_to_reset
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self._full = False
    
    def store(self, obs, act, rew, next_obs, done):
        if self._ready_to_reset:
            self.reset()
            self._ready_to_reset = False
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        if done:
            self.n_step_returns = calculate_nstep_returns(self.rew_buf, self.n, self.gamma)
            self._full = True
        return self._full
    
    def get_trajectory(self):
        assert self._full
        self._ready_to_reset = True
        return self.obs_buf, self.act_buf, self.n_step_returns, self.done_buf

        
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs.squeeze()
        self.obs2_buf[self.ptr] = next_obs.squeeze()
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class MultiStepReplayBuffer(ReplayBuffer):
    """
    A FIFO experience replay buffer for TD3 agents with Multi-step learning.
    Important: Sampling from this buffer only works after the first episode has been processed.
    """
    def __init__(self, obs_dim, act_dim, size, n, gamma):
        super().__init__(obs_dim, act_dim, size)
        self.intermediate_buffer = MultiStepIntermediateBuffer(n, gamma)
        self.n = n

    def store(self, obs, act, rew, next_obs, done):
        intermediate_is_full = self.intermediate_buffer.store(obs, act, rew, next_obs, done)
        if intermediate_is_full:
            obss, acts, n_step_returns, dones = self.intermediate_buffer.get_trajectory()
            for i, (obs, act, n_step_return, done) in enumerate(zip(obss, acts, n_step_returns, dones)):
                self.obs_buf[self.ptr] = obs.squeeze()
                self.act_buf[self.ptr] = act
                self.rew_buf[self.ptr] = n_step_return
                if i + self.n < len(obss):
                    self.obs2_buf[self.ptr] = obss[i + self.n].squeeze()
                    self.done_buf[self.ptr] = dones[i + self.n]
                else:
                    self.obs2_buf[self.ptr] = obs.squeeze() # Wrong but doesn't matter because multiplied by 0 anyway
                    self.done_buf[self.ptr] = True
                self.ptr = (self.ptr+1) % self.max_size
                self.size = min(self.size+1, self.max_size)