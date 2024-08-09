import pickle
# import matplotlib.pyplot as plt
import numpy as np

# 1. 从 .pkl 文件中加载数据
with open('/home/foamlab/nw/rosbags/cylinder/pkls/rosbag2_2024_08_01-16_53_28.pkl', 'rb') as file:
    data = pickle.load(file)

# 2. 打印字典的键和示例数据
print("字典的键:", data.keys())
for key in data.keys():
    print(f"键: {key}, 数据类型: {type(data[key])}, 形状: {np.array(data[key]).shape}")

# 3. 打印 motor_cmd 和 joint_state 的前几个元素
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

# 4. 根据数据结构调整可视化代码
# 假设图像数据存储在键 'images' 下
# if 'image' in data:
#     images = data['image']
    
#     # 确保 images 是一个可视化的格式
#     if isinstance(images, np.ndarray) and images.ndim == 4:
#         num_images = images.shape[0]
#         fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#         for i in range(num_images):
#             axes[i].imshow(images[i])
#             axes[i].axis('off')
#         plt.show()
#     elif isinstance(images, list):
#         num_images = len(images)
#         fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#         for i in range(num_images):
#             axes[i].imshow(images[i])
#             axes[i].axis('off')
#         plt.show()
#     else:
#         print("Unsupported images format within dictionary")
# else:
#     print("键 'image' 不存在")



# import pickle
# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 从 .pkl 文件中加载数据
# with open('70.pkl', 'rb') as file:
#     data = pickle.load(file)

# # 2. 打印字典的键和示例数据
# print("字典的键:", data.keys())
# for key in data.keys():
#     print(f"键: {key}, 数据类型: {type(data[key])}, 形状: {np.array(data[key]).shape}")

# # 3. 打印 motor_cmd 和 joint_state 的前几个元素
# print("\ndeltaJointState 的前5个元素:")
# for i in range(5):
#     print(f"第{i+1}个元素:", data['deltaJointState'][i])

# print("\ndeltaControl 的前5个元素:")
# for i in range(5):
#     print(f"第{i+1}个元素:", data['deltaControl'][i])

# # 4. 根据数据结构调整可视化代码
# # 假设图像数据存储在键 'images' 下
# if 'externalImg' in data:
#     images = data['externalImg']
    
#     # 确保 images 是一个可视化的格式
#     if isinstance(images, np.ndarray) and images.ndim == 4:
#         num_images = images.shape[0]
#         fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#         for i in range(num_images):
#             axes[i].imshow(images[i])
#             axes[i].axis('off')
#         plt.show()
#     elif isinstance(images, list):
#         num_images = len(images)
#         fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#         for i in range(num_images):
#             axes[i].imshow(images[i])
#             axes[i].axis('off')
#         plt.show()
#     else:
#         print("Unsupported images format within dictionary")
# else:
#     print("键 'image_data' 不存在")
