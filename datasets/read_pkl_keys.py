import pickle
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data from a .pkl file
with open('/home/foamlab/nw/mydata/cylinder/pkls/rosbag2_2024_08_17-10_57_39.pkl', 'rb') as file:
    data = pickle.load(file)

# 2. Print the keys of the dictionary and sample data
print("Keys of the dictionary:", data.keys())
for key in data.keys():
    print(f"Key: {key}, Data type: {type(data[key])}, Shape: {np.array(data[key]).shape}")

# 3. Print the first few elements of motor_cmd and joint_state
print("\n/autohand_node/cmd_autohand sample elements:")
for i in range(5):
    print(f"Element {i+1}:", data['cmd_hand'][i])

print("\n/xarm/move_joint_cmd sample elements:")
for i in range(50):
    print(f"Element {i+1}:", data['cmd_xarm'][i])

print("\n/autohand_node/state_autohand sample elements:")
for i in range(5):
    print(f"Element {i+1}:", data['state_hand'][i])

print("\n/xarm/joint_states sample elements:")
for i in range(5):
    print(f"Element {i+1}:", data['state_xarm'][i])

# 4. Adjust visualization code based on the data structure
# Assuming image data is stored under the key 'images'
if 'image' in data:
    images = data['image']
    
    # Ensure images are in a visualizable format
    if isinstance(images, np.ndarray) and images.ndim == 4:
        num_images = images.shape[0]
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis('off')
        plt.show()
    elif isinstance(images, list):
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis('off')
        plt.show()
    else:
        print("Unsupported images format within dictionary")
else:
    print("Key 'image' does not exist")
