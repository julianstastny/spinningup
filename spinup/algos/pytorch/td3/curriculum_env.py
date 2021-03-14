from laserhockey.hockey_env import HockeyEnv
from laserhockey.hockey_env import BasicOpponent
from gym import spaces
import numpy as np
import torch
import random


class CurriculumEnv(HockeyEnv):
  
  def __init__(self, mode=None, weak_opponent=True, self_play_path=None, num_saved_models=1):
    self.opponent = BasicOpponent(weak=weak_opponent)
    self.episode = 0
    self.curriculum = {
        "defense": 0,
        "shooting": 0,
        "defense+shooting": 2000, # Until episode 1000
        "weak opp": [], #list(range(1000))
        "self-play": 6000
    }
    super().__init__(mode=1, keep_mode=True)
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
    self.self_play_path = self_play_path
    self.num_saved_models = num_saved_models
    assert not (self_play_path is None) and self.curriculum["self-play"]


  def reset(self, one_starting=None, opponent="default", mode="default"):
    self.episode += 1

    # First, set the strength of the opponent.
    if (self.episode in self.curriculum["weak opp"]) or (opponent=="weak"):
      self.opponent = BasicOpponent(weak=True)
      self.opponent_act = lambda obs: self.opponent.act(obs)
      stage = "wb"
    elif (self.episode < self.curriculum["self-play"]) or (opponent=="strong"):
      self.opponent = BasicOpponent(weak=False)
      self.opponent_act = lambda obs: self.opponent.act(obs)
      stage = "sb"
    else:
      which_model = random.randint(1, self.num_saved_models)
      self.self_play_path = self.self_play_path[:-4] + str(which_model) + ".pt"
      try:
        self.opponent_model = torch.load(self.self_play_path)
      except: # In case we haven't gotten as far wrt saving models
        self.self_play_path = self.self_play_path[:-4] + str(1) + ".pt"
        self.opponent_model = torch.load(self.self_play_path)
 
      self.opponent_act = lambda obs: self.opponent_model.act(torch.as_tensor(obs, dtype=torch.float32))
      stage = "sp"

    # Second, decide which mode to play in.
    if (self.episode < self.curriculum["defense"]) or (mode==2):
      stage += "D"
      obs = super().reset(one_starting, mode=2) 
    elif (self.episode < self.curriculum["shooting"]) or (mode==1):
      stage += "S"
      obs = super().reset(one_starting, mode=1) 
    elif (self.episode < self.curriculum["defense+shooting"]) and (mode=="default"):
      stage += "DS"
      if self.episode % 2 == 0:
        obs = super().reset(one_starting, mode=1)
      else: 
        obs = super().reset(one_starting, mode=2)
    else:
      stage += "P"
      obs = super().reset(one_starting, mode=0)
    self.stage = stage
    return obs

  def step(self, action):
    with torch.no_grad():
      ob2 = self.obs_agent_two()
      # a2 = self.opponent.act(ob2)
      a2 = self.opponent_act(ob2)
      action2 = np.hstack([action, a2])
      obs, reward, done, info = super().step(action2)
      info["stage"] = self.stage
      if self.mode != 1:
        reward -= info["reward_closeness_to_puck"]
    return obs, float(reward), done, info


from gym.envs.registration import register

try:
  register(
    id='HockeyCurriculum-v0',
    entry_point='curriculum_env:CurriculumEnv',
    kwargs={'mode': 0}
  )
except Exception as e:
  print(e)