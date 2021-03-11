from spinup.utils.test_policy import load_policy_and_env, run_policy
import gym
import custompendulumenv
import laserhockey

_, get_action = load_policy_and_env('/Users/julianstastny/Code/rl-course/Hockey-project/spinningup/data/td3-layernorm-all/td3-layernorm-all_s0')
env = gym.make('CustomPendulum-v0')

#_, get_action = load_policy_and_env('/path/to/output_directory')
#env = gym.make('CustomPendulum-v0')

run_policy(env, get_action)