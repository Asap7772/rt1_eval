import numpy as np
from collections import defaultdict

from tf_agents.trajectories import time_step as ts
from tf_agents.policies import py_tf_eager_policy

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import os 
import pickle 
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from collections import defaultdict
from gym.spaces import Box

def stack_observations(observations):
    print('keys', observations[0].keys())
    new_dict = defaultdict(lambda: [])
    
    for key in observations[0]:
        for i in range(len(observations)):
            new_dict[key].append(observations[i][key])
    
    stacked_dict = {}
    for key in new_dict:
        stacked_dict[key] = np.stack(new_dict[key])
    
    del new_dict
    
    return stacked_dict


def unnormalize_actions(actions):
    action_space = Box(np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]), np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]), dtype=np.float32)
    low, high = action_space.low, action_space.high
    
    actions = actions.squeeze()
    actions_rescaled = (actions + 1) / 2 * (high - low) + low

    if np.any(actions_rescaled > high):
        print('action bounds violated: ', actions)
    if np.any(actions_rescaled < low):
        print('action bounds violated: ', actions)
    return actions_rescaled

def normalize_task_name(task_name):

  replaced = task_name.replace('_', ' ').replace('1f', ' ').replace(
      '4f', ' ').replace('-', ' ').replace('50',
                                           ' ').replace('55',
                                                        ' ').replace('56', ' ')
  return replaced.lstrip(' ').rstrip(' ')

def tfa_action_to_bridge_action(tfa_action):
  return np.concatenate((tfa_action['world_vector'], tfa_action['rotation_delta'], tfa_action['gripper_closedness_action']))

class RT1WrapperPolicy():
    def __init__(self, saved_model_path='/home/deepthought/interbotix_ws/src/RT-1 on Bridge/000073080', task='put_pot_in_sink'):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
        # Invalid device or cannot modify virtual devices once initialized.
            pass

        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True)
        
        self.base_observation = {
            'image':
                np.zeros(shape=(256, 320, 3), dtype=np.uint8),
            'natural_language_embedding':
                np.zeros(shape=(512), dtype=np.float32),
            'gripper_closed':
                np.zeros(shape=(1), dtype=np.float32),
            'height_to_bottom':
                np.zeros(shape=(1), dtype=np.float32),
            'base_pose_tool_reached':
                np.zeros(shape=(7), dtype=np.float32),
            'workspace_bounds':
                np.zeros(shape=(3, 3), dtype=np.float32),
            'orientation_box':
                np.zeros(shape=(2, 3), dtype=np.float32),
            'orientation_start':
                np.zeros(shape=(4), dtype=np.float32),
            'src_rotation':
                np.zeros(shape=(4), dtype=np.float32),
            'robot_orientation_positions_box':
                np.zeros(shape=(3, 3), dtype=np.float32),
            'natural_language_instruction':
                np.zeros(shape=(), dtype=str),
            'vector_to_go':
                np.zeros(shape=(3), dtype=np.float32),
            'rotation_delta_to_go':
                np.zeros(shape=(3), dtype=np.float32),
            'gripper_closedness_commanded':
                np.zeros(shape=(1), dtype=np.float32),
        }
        
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
        self.natural_language_embedding = embed([normalize_task_name(task)])[0]
        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)
    
    def forward(self, observations):
        
        if isinstance(observations, list):
            image = observations[-1]['pixels']
        else:
            assert False, 'needs to be sequence'
        
        
        image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
        image = tf.cast(image, np.uint8)
        assert image.shape == (1, 256,320, 3), f"image shape is {image.shape}"
        
        self.base_observation['image'] = image.numpy().squeeze()
        self.base_observation['natural_language_embedding'] = self.natural_language_embedding

        
        tfa_time_step = ts.transition(self.base_observation, reward=np.zeros(()))
        
        policy_step = self.tfa_policy.action(tfa_time_step, self.policy_state)
        action = policy_step.action
        self.policy_state = policy_step.state
        
        unnorm_action = tfa_action_to_bridge_action(action)
        action = unnormalize_actions(unnorm_action)
        
        print('action', action, np.linalg.norm(action, 2))
        print('unnorm_action', unnorm_action, np.linalg.norm(unnorm_action, 2))
        
        return action
        
        
        