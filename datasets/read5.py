import pickle
# import matplotlib.pyplot as plt
import math
import numpy as np
with open('/home/foamlab/nw/rosbags/cylinder/pkls/rosbag2_2024_08_01-16_52_56.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())
