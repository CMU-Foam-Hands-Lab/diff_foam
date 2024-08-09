import pickle
# import matplotlib.pyplot as plt
import numpy as np

# 1. 从 .pkl 文件中加载数据
with open('/home/foamlab/nw/rosbags/cylinder/pkls/rosbag2_2024_08_01-16_55_54.pkl', 'rb') as file:
    data = pickle.load(file)

# 2. 打印字典的键和示例数据
print("字典的键:", data.keys())
for key in data.keys():
    print(f"键: {key}, 数据类型: {type(data[key])}, 形状: {np.array(data[key]).shape}")

# # 3. 根据数据结构调整可视化代码
# # 假设图像数据存储在键 'images' 下
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


"""
字典的键: dict_keys(['image_data'])
键: image_data, 数据类型: <class 'list'>, 形状: (108, 240, 320, 3)
字典 data 中包含一个键 'image_data'，其对应的值是一个形状为 (108, 240, 320, 3) 的列表。
这个数据结构表示有 108 张图像，每张图像的大小为 240x320 像素，具有 3 个颜色通道（RGB）。
"""