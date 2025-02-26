from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
import custompendulumenv
from replay_buffers import ReplayBuffer, MultiStepReplayBuffer
import os


def td3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=100, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, multistep_n=1, use_parameter_noise=False,
        decay_exploration=False, save_k_latest=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    sigma = act_noise


    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    if multistep_n > 1:
        replay_buffer = MultiStepReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, n=multistep_n, gamma=gamma)
    else:
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma**(multistep_n) * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        noise_dim = None if act_dim==1 else act_dim # Fixes a bug that occurs in case of 1-dimensional action spaces
        a += noise_scale * np.random.standard_normal(noise_dim)
        return np.clip(a, -act_limit, act_limit)

    def get_action_from_perturbed_model(o, actor):
        a = actor.act(torch.as_tensor(o, dtype=torch.float32))
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            if type(env).__class__.__name__ == 'CurriculumEnv':
                o, d, ep_ret, ep_len = test_env.reset(opponent="strong", mode=0), False, 0, 0
            else:
                o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
 
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        return ep_ret

    def get_perturbed_model(sigma):
        actor = deepcopy(ac)
        with torch.no_grad():
            for param in actor.pi.parameters():
                param.add_(torch.randn(param.size()) * sigma)
        return actor

    def update_sigma(target_noise, sigma, actor, perturbed_actor, batch_size=128):
        obs = replay_buffer.sample_batch(batch_size=batch_size)['obs']
        with torch.no_grad():
            ac1 = actor.pi(obs)
            ac2 = perturbed_actor.pi(obs)
            dist = torch.sqrt(torch.mean((ac1 - ac2)**2))
        if dist < target_noise:
            sigma *= 1.01
        else:
            sigma /= 1.01
        return sigma


    # Prepare for interaction with environment
    best_test_ep_return = float('-inf')
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    save_id = 0


    if use_parameter_noise:
        perturbed_actor = get_perturbed_model(sigma)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            if use_parameter_noise:
                a = get_action_from_perturbed_model(o.squeeze(), perturbed_actor)
            else:
                a = get_action(o.squeeze(), act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0


        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)
                if use_parameter_noise:
                    perturbed_actor = get_perturbed_model(sigma) # To make sure that perturbed actor is based on the same actor that we are testing against
                    sigma = update_sigma(act_noise, sigma, ac, perturbed_actor)
                    perturbed_actor = get_perturbed_model(sigma)


        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:

             # Decay the action noise or sigma
            if decay_exploration and (((t+1) % steps_per_epoch) == 0):
                sigma /= 1.001
                act_noise /= 1.001
                
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_ep_return = test_agent()

            # Save best model:
            if test_ep_return > best_test_ep_return:
                best_test_ep_return = test_ep_return
                logger.save_state({'env': env}, 0) 

            # Save latest model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            save_id += 1
            logger.save_state({'env': env}, save_id % save_k_latest)


            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            if use_parameter_noise:
                logger.log_tabular('sigma', sigma)
            if 'stage' in info.keys():
                logger.log_tabular('CStage', info['stage'])
            logger.dump_tabular()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hockey-One-v0')
    parser.add_argument('--weak_opponent', type=int, default=1)
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--psn', type=int, default=0) # Will be converted to boolean
    parser.add_argument('--decay', type=int, default=0) # Will be converted to boolean
    parser.add_argument('--layernorm', type=int, default=1) # Will be converted to boolean
    parser.add_argument('--data_path', type=str, default='/Users/julianstastny/Code/rl-course/Hockey-project/spinningup/data')
    parser.add_argument('--save_k_latest', type=int, default=1)
    parser.add_argument('--self_play', type=int, default=1)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    if "Hockey" in args.env:
        import laserhockey
        import curriculum_env
        more_detailed = "Curr" if "Curriculum" in args.env else f"{args.env}m{args.mode}o{args.weak_opponent}"
        experiment_name = f"{args.exp_name}_{more_detailed}_{args.l}x{args.hid}_psn{args.psn}_decay{args.decay}_nstep{args.n}_ln{args.layernorm}"
        logger_kwargs = setup_logger_kwargs(experiment_name, args.seed)
        if args.self_play:
            self_play_path = os.path.join(args.data_path, experiment_name, experiment_name + '_s' + str(args.seed), "pyt_save/model0.pt")
            print(self_play_path)
        else:
            self_play_path = None
            print("Curriculum without self-play")
        
        if "Curriculum" in args.env:
            env_init = lambda : gym.make(args.env, mode=args.mode, weak_opponent=bool(args.weak_opponent), self_play_path=self_play_path, num_saved_models=args.save_k_latest)
        else:
            env_init = lambda : gym.make(args.env, mode=args.mode, weak_opponent=bool(args.weak_opponent)) 

        td3(env_init, actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, layernorm=args.layernorm), 
            gamma=args.gamma, seed=args.seed, epochs=args.epochs,
            logger_kwargs=logger_kwargs, multistep_n=args.n, use_parameter_noise=bool(args.psn),
            decay_exploration=bool(args.decay), save_k_latest=args.save_k_latest)
    else:
        experiment_name = f"{args.exp_name}_{args.env}_{args.l}x{args.hid}_psn{args.psn}_decay{args.decay}_nstep{args.n}_ln{args.layernorm}"
        logger_kwargs = setup_logger_kwargs(experiment_name, args.seed)
 
        td3(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
            gamma=args.gamma, seed=args.seed, epochs=args.epochs,
            logger_kwargs=logger_kwargs, multistep_n=args.n, use_parameter_noise=bool(args.psn),
            decay_exploration=bool(args.decay))
