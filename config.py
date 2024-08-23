import numpy as np

# dataset parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

# network parameters
# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 23 dimensional
lowdim_obs_dim = 23
# observation feature has 512 + 23 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
# agent_action is 23 dimensional
action_dim = 23

min_values = np.array([0.000] * 23)
max_values = np.array([1.000] * 23)
min_values[3] = -1.000  
max_values[3] = 1.000  
min_values[11] = -1.000 
max_values[11] = 1.000 
min_values[16:] = np.array([-0.28, -0.78, -1.19, 0.13, -0.15, 0.14, -2.79])
max_values[16:] = np.array([0.66,  0.20,  0.17,  1.67, 1.06,  1.68, -0.71])

stats = {
'min': min_values,
'max': max_values
}