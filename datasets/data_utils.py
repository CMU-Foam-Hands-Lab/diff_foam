import numpy as np
import math
import torch
import pickle
import os
import torchvision.transforms as transforms

image_transforms = transforms.Compose([transforms.CenterCrop(size=(216,288))])

def normalize_data(data, stats):
    ndata = np.zeros_like(data)
    
    # Normalize first 16 dimensions
    ndata[:, [3, 11]] = data[:, [3, 11]]
    
    # Normalize the other 14 dimensions to [-1, 1]
    for i in range(16):
        if i not in [3, 11]:  # dimensions 4 and 12
            ndata[:, i] = (data[:, i] * 2) - 1  # Mapping from [0, 1] to [-1, 1]
    
    # Normalize the last 7 dimensions
    last_7_start = 16
    last_7_end = 23
    
    for i in range(last_7_start, last_7_end):
        range_ = stats['max'][i] - stats['min'][i]
        ndata[:, i] = 2 * (data[:, i] - stats['min'][i]) / range_ - 1  # Mapping from [min, max] to [-1, 1]
    
    return ndata

def unnormalize_data(ndata, stats):
    data = np.zeros_like(ndata)
    
    # Unnormalize first 16 dimensions
    data[:, [3, 11]] = ndata[:, [3, 11]]
    
    # Unnormalize the other 14 dimensions from [-1, 1] to [0, 1]
    for i in range(16):
        if i not in [3, 11]:  # dimensions 4 and 12
            data[:, i] = (ndata[:, i] + 1) / 2  # Mapping from [-1, 1] to [0, 1]
    
    # Unnormalize the last 7 dimensions
    last_7_start = 16
    last_7_end = 23
    
    for i in range(last_7_start, last_7_end):
        range_ = stats['max'][i] - stats['min'][i]
        data[:, i] = (ndata[:, i] + 1) / 2 * range_ + stats['min'][i]  # Restore to original range
    
    return data

def normalize_images(images):
    # resize image to (120, 160)
    # nomalize to [0,1]
    nimages = images / 255.0
    return nimages
