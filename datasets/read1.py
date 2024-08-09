import pickle

# 打开 .pkl 文件
with open('rosbag2_2024_08_01-16_48_43.pkl', 'rb') as file:
    # 读取数据
    data = pickle.load(file)

# 打印数据
print(data)
