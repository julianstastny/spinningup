from laserhockey.hockey_env import HockeyEnv
from laserhockey.hockey_env import BasicOpponent
from gym import spaces
import numpy as np


class CurriculumEnv(HockeyEnv):
  
  def __init__(self, mode=None):
    self.opponent = None
    self.episode = 0
    self.curriculum = {
        "defense+shooting": 1000,
        "weak": 1500
    }
    super().__init__(mode=1, keep_mode=True)
    # linear force in (x,y)-direction, torque, and shooting
    # self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)


  def reset(self, one_starting=None):
    self.episode += 1
    if self.episode < self.curriculum["defense+shooting"]:
        print("training defense and shooting")
        if self.episode % 2 == 0:
            return super().reset(one_starting, mode=1)
        return super().reset(mode=2)
    elif self.episode < self.curriculum["weak"]:
        self.opponent = BasicOpponent(weak=True)
    # elif self.episode < 2000:
    else:
        self.opponent = BasicOpponent(weak=False)
    return super().reset(one_starting, mode=0)


  def step(self, action):
    if self.opponent is None:
        return super().step(action)
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