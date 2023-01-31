from email.policy import default
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

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

class BETWrapperPolicy(nn.Module):
    def __init__(self, models, num_tasks=108):
        super().__init__()
        self.obs_encoding_net = models['obs_encoding_net']
        self.action_ae = models['action_ae']
        self.state_prior = models['state_prior']
        self.task_tensor = torch.zeros((1, 1, num_tasks)).cuda().float()
        self.task_tensor[:, :, -1] = 0
    
    def forward(self, observations):
        with torch.no_grad():
            if isinstance(observations, list):
                observations = stack_observations(observations)
                norm_image = observations['pixels']/255.
                norm_image = np.transpose(norm_image, (0, 1, 4, 2, 3))
                observation = torch.from_numpy(norm_image).cuda().float()
            else:
                assert False, 'needs to be sequence'
            
            obs_encoding = self.obs_encoding_net(observation)
            
            obs_encoding = torch.cat([obs_encoding, self.task_tensor], dim=-1)
            obs_encoding = self.obs_encoding_net.linear_proj(obs_encoding)
            
            latents = self.state_prior.generate_latents(obs_encoding, None)
            actions = self.action_ae.decode_actions(latents, obs_encoding)
            
            return actions.squeeze().cpu().detach().numpy() # (action_dim,)
        
        
    