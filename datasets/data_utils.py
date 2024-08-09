import numpy as np
import math
import torch
import pickle
import os
import torchvision.transforms as transforms

image_transforms = transforms.Compose([transforms.CenterCrop(size=(216,288))])

def normalize_data(data, stats):
    ndata = np.copy(data)
    norm_range = stats['max'] - stats['min']
    indices_to_normalize = [i for i in range(23) if i not in [3, 11]]
    for i in indices_to_normalize:
        ndata[i] = (data[i] - stats['min'][i]) / norm_range[i] * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    data = np.copy(ndata)
    indices_to_normalize = [i for i in range(23) if i not in [3, 11]]
    for i in indices_to_normalize:
        ndata[i] = (ndata[i] + 1) / 2
    norm_range = stats['max'] - stats['min']
    for i in indices_to_normalize:
        data[i] = ndata[i] * norm_range[i] + stats['min'][i]
    return data

def normalize_images(images):
    # resize image to (120, 160)
    # nomalize to [0,1]
    nimages = images / 255.0
    return nimages
