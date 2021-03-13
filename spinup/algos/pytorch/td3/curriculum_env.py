from laserhockey.hockey_env import HockeyEnv
from laserhockey.hockey_env import BasicOpponent
from gym import spaces
import numpy as np


class CurriculumEnv(HockeyEnv):
  
  def __init__(self, mode=None, weak_opponent=True):
    self.opponent = BasicOpponent(weak=weak_opponent)
    self.episode = 0
    self.curriculum = {
        "defense": 0,
        "shooting": 0,
        "defense+shooting": 2000, # Until episode 1000
        "weak opp": []#list(range(1000))
    }
    super().__init__(mode=1, keep_mode=True)
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)


  def reset(self, one_starting=None):
    self.episode += 1

    # First, set the strength of the opponent.
    if self.episode in self.curriculum["weak opp"]:
      self.opponent = BasicOpponent(weak=True)
      stage = "w"
    else:
      self.opponent = BasicOpponent(weak=False)
      stage = "s"

    # Second, decide which mode to play in.
    if self.episode < self.curriculum["defense"]:
      stage += "D"
      obs = super().reset(one_starting, mode=2) 
    elif self.episode < self.curriculum["shooting"]:
      stage += "S"
      obs = super().reset(one_starting, mode=1) 
    elif self.episode < self.curriculum["defense+shooting"]:
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
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
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