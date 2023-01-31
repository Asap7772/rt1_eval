import argparse
from email.policy import default
from typing import DefaultDict
import torch
import numpy as np
from collections import defaultdict
from widowx_real_env import *
from bet_wrapper import BETWrapperPolicy
import time

TARGET_POINT = np.array([0.28425417, 0.04540814, 0.07545623])  # mean

class AttrDict(defaultdict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

variant=AttrDict(lambda: False)

def get_env_params(start_transform):
    env_params = {
        'fix_zangle': True,  # do not apply random rotations to start state
        'move_duration': 0.2,
        'adaptive_wait': True,
        'move_to_rand_start_freq': 1,
        'override_workspace_boundaries': [[0.100, -0.25, 0.0, -1.57, 0], [0.41, 0.143, 0.33, 1.57, 0]],

        'action_clipping': 'xyz',
        'catch_environment_except': True,
        'target_point': TARGET_POINT,
        'add_states': variant.add_states,
        'from_states': variant.from_states,
        'reward_type': variant.reward_type,
        'start_transform': None if variant.start_transform == '' else start_transforms[start_transform],
        'randomize_initpos': 'full_area'
    }
    return env_params

def eval(policy, env, num_episodes=1):
    """
    Evaluate a policy.
    :param policy: policy to be evaluated
    :param env: environment to be evaluated on
    :param num_episodes: number of episodes to run
    :param render: whether to render the environment
    :return: mean reward
    """
    rewards = []
    obs_so_far = []
    
    for i in range(num_episodes):
        env.start()  # this sets the correct moving time for the robot
        last_tstep = time.time()
        step_duration = 0.2
        
        obs, done = env.reset(), False
        total_reward = 0.0
        
        print(f'Episode: {i}')       
        t = 0
        while not done:
            obs_so_far.append(obs)
            if time.time() > last_tstep + step_duration:
                if (time.time() - last_tstep) > step_duration * 1.05:
                    print('###########################')
                    print('Warning, loop takes too long: {}s!!!'.format(time.time() - last_tstep))
                    print('###########################')
                if (time.time() - last_tstep) < step_duration * 0.95:
                    print('###########################')
                    print('Warning, loop too short: {}s!!!'.format(time.time() - last_tstep))
                    print('###########################')
                
                last_tstep = time.time()
                print(t)
                t+=1
            
                action = policy(obs_so_far)
                tstamp_return_obs = last_tstep + step_duration
                obs, reward, done, _ = env.step({'action':action, 'tstamp_return_obs':tstamp_return_obs})
                total_reward += reward
                rewards.append(total_reward)   
                
    return torch.mean(torch.tensor(rewards))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, default='/home/datacol1/interbotix_ws/src/robonetv2/code/bet/data/exp_local/2022.08.31/091404_bridge_data/snapshot.pt')
    argparser.add_argument('--start_transform', type=str, default='toykitchen6_croissant_pot_cardboard_sampled')
    argparser.add_argument('--num_tasks', type=int, default=0)
    argparser.add_argument('--num_trajectory', type=int, default=10)
    args = argparser.parse_args()
    
    new_path = args.path
    if '/home/datacol1/interbotix_ws/src/robonetv2' in args.path:
        new_path = new_path.replace('/home/datacol1/interbotix_ws/src/robonetv2', '/home/robonetv2')
    
    print(f'loading {new_path}')
    model = torch.load(new_path)
    policy = BETWrapperPolicy(model)
    
    from widowx_real_env import JaxRLWidowXEnv
    env_params = get_env_params(args.start_transform)
    env = JaxRLWidowXEnv(env_params, num_tasks=args.num_tasks)
    
    eval(policy, env, num_episodes=args.num_trajectory)