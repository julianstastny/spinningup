from laserhockey.hockey_env import HockeyEnv
from laserhockey.hockey_env import BasicOpponent
from gym import spaces
import numpy as np


class CurriculumEnv(HockeyEnv):
  
  def __init__(self, mode=None):
    self.opponent = BasicOpponent(weak=True)
    self.episode = 0
    self.curriculum = {
        "defense+shooting": 1000, # Until episode 1000
        "play vs weak": 2000 # Until episode 2000. After that, play vs strong
    }
    super().__init__(mode=1, keep_mode=True)
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)


  def reset(self, one_starting=None):
    self.episode += 1
    if self.episode < self.curriculum["defense+shooting"]:
        if self.episode % 2 == 0:
            return super().reset(one_starting, mode=1)
        return super().reset(mode=2)
    elif self.episode > self.curriculum["play vs weak"]:
        self.opponent = BasicOpponent(weak=False)
    return super().reset(one_starting, mode=0)


  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)


from gym.envs.registration import register

try:
  register(
    id='HockeyCurriculum-v0',
    entry_point='curriculum_env:CurriculumEnv',
    kwargs={'mode': 0}
  )
except Exception as e:
  print(e)