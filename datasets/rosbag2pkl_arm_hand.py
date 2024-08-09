import os
from os import path as osp
import cv2
import rosbag2_py
import numpy as np
import pickle as pkl
from cv_bridge import CvBridge
import argparse
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

class RosbagReader:
    def __init__(self, bag_path, file_name, save_path, file_time):
        self.bag_path = osp.join(bag_path, file_name)
        self.file_name = file_name[:-4] if file_name.endswith('.bag') else file_name
        self.file_time = file_time
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.bridge = CvBridge()
        self.downsample_rate = 3

        try:
            self.bag = rosbag2_py.SequentialReader()
            self.bag.open(storage_options, converter_options)
        except Exception as e:
            print(f"Failed to open bag file: {e}")
            self.bag = None

        self.cmd_hand_data = []
        self.cmd_hand_time = []
        self.cmd_xarm_data = []
        self.cmd_xarm_time = []

        self.state_hand_data = []
        self.state_hand_time = []
        self.state_xarm_data = []
        self.state_xarm_time = []

        self.image_data = []
        self.image_time = []

    def decode_joint_state(self, msg):
        try:
            return msg.position
        except Exception as e:
            print(f"Error decoding JointState message: {e}")
            return []

    def read_data(self):
        if self.bag is None:
            print("Bag file is not initialized.")
            return

        count_cmd_hand = 0
        count_cmd_xarm = 0
        count_state_hand = 0
        count_state_xarm = 0
        count_image = 0

        joint_state_type = get_message('sensor_msgs/msg/JointState')

        cmd_hand_file_path = osp.join(self.save_path, 'cmd_hand.txt')
        cmd_xarm_file_path = osp.join(self.save_path, 'cmd_xarm.txt')
        state_hand_file_path = osp.join(self.save_path, 'state_hand.txt')
        state_xarm_file_path = osp.join(self.save_path, 'state_xarm.txt')

        # Ensure that the file exists, or create it if it does not
        open(cmd_hand_file_path, 'a').close()
        open(cmd_xarm_file_path, 'a').close()
        open(state_hand_file_path, 'a').close()
        open(state_xarm_file_path, 'a').close()

        with open(cmd_hand_file_path, 'a') as f1, \
            open(cmd_xarm_file_path, 'a') as f2, \
            open(state_hand_file_path, 'a') as f3, \
            open(state_xarm_file_path, 'a') as f4:
            while self.bag.has_next():
                topic, msg, t = self.bag.read_next()

                if topic == '/autohand_node/cmd_autohand':
                    try:
                        msg = deserialize_message(msg, joint_state_type)
                        data = self.decode_joint_state(msg)
                        self.cmd_hand_data.append(data)
                        self.cmd_hand_time.append(t)
                        count_cmd_hand += 1
                        f1.write(f"Time: {t}, Data: {data}\n")
                    except Exception as e:
                        print(f"Error decoding /autohand_node/cmd_autohand message: {e}")

                elif topic == '/xarm/move_joint_cmd':
                    try:
                        msg = deserialize_message(msg, joint_state_type)
                        data = self.decode_joint_state(msg)
                        self.cmd_xarm_data.append(data)
                        self.cmd_xarm_time.append(t)
                        count_cmd_xarm += 1
                        f2.write(f"Time: {t}, Data: {data}\n")
                    except Exception as e:
                        print(f"Error decoding /xarm/move_joint_cmd message: {e}")

                elif topic == '/autohand_node/state_autohand':
                    if count_state_hand % self.downsample_rate == 0:
                        try:
                            msg = deserialize_message(msg, joint_state_type)
                            data = self.decode_joint_state(msg)
                            self.state_hand_data.append(data)
                            self.state_hand_time.append(t)
                            f3.write(f"Time: {t}, Data: {data}\n")
                        except Exception as e:
                            print(f"Error decoding /autohand_node/state_autohand message: {e}")
                    count_state_hand += 1

                elif topic == '/xarm/joint_states':
                    try:
                        msg = deserialize_message(msg, joint_state_type)
                        data = self.decode_joint_state(msg)
                        self.state_xarm_data.append(data)
                        self.state_xarm_time.append(t)
                        count_state_xarm += 1
                        f4.write(f"Time: {t}, Data: {data}\n")
                    except Exception as e:
                        print(f"Error decoding /xarm/joint_states message: {e}")

                elif topic == '/camera/color/image_raw/compressed' and isinstance(msg, bytes):
                    try:
                        start_idx = msg.find(b'\xff\xd8')
                        end_idx = msg.find(b'\xff\xd9')

                        if start_idx == -1 or end_idx == -1:
                            print(f"JPEG markers not found. Start index: {start_idx}, End index: {end_idx}")
                            continue

                        jpeg_data = msg[start_idx:end_idx + 2]
                        np_arr = np.frombuffer(jpeg_data, np.uint8)
                        cur_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if cur_image is None:
                            print("Failed to decode image")
                            print(f"JPEG data length: {len(jpeg_data)}")
                            continue

                        cur_image = cv2.resize(cur_image, (320, 240))
                        self.image_data.append(cur_image)
                        self.image_time.append(t)
                        count_image += 1
                    except Exception as e:
                        print(f"Error processing /camera/color/image_raw/compressed message: {e}")

        print("# /autohand_node/cmd_autohand data: ", count_cmd_hand)
        print("# /xarm/move_joint_cmd data: ", count_cmd_xarm)
        print("# /autohand_node/state_autohand data: ", count_state_hand)
        print("# /xarm/joint_states data: ", count_state_xarm)
        print("# /camera/color/image_raw/compressed data: ", count_image)

    def save_pkl(self):
        if self.bag is None:
            print("Bag file is not initialized.")
            return

        data_filename = osp.join(self.save_path, self.file_name + ".pkl")
        print("File name: ", self.file_name + ".pkl")

        pkl_data = {
            'cmd_hand': [],
            'cmd_xarm': [],
            'state_hand': [],
            'state_xarm': [],
            'image': []
        }

        cmd_hand_time = np.array(self.cmd_hand_time)
        cmd_xarm_time = np.array(self.cmd_xarm_time)
        state_hand_time = np.array(self.state_hand_time)
        state_xarm_time = np.array(self.state_xarm_time)
        image_time = np.array(self.image_time)

        synced_idx0 = []
        synced_idx1 = []
        synced_idx2 = []
        synced_idx3 = []

        count_cmd_hand = 0
        count_cmd_xarm = 0
        count_state_arm = 0
        count_image = 0

        for i in range(len(state_hand_time)):
            while count_image < len(image_time) and image_time[count_image] < state_hand_time[i]:
                count_image += 1
            synced_idx0.append(count_image)

            while count_cmd_hand < len(cmd_hand_time) and cmd_hand_time[count_cmd_hand] < state_hand_time[i]:
                count_cmd_hand += 1
            synced_idx1.append(count_cmd_hand)

            while count_cmd_xarm < len(cmd_xarm_time) and cmd_xarm_time[count_cmd_xarm] < state_hand_time[i]:
                count_cmd_xarm += 1
            synced_idx2.append(count_cmd_xarm)

            while count_state_arm < len(state_xarm_time) and state_xarm_time[count_state_arm] < state_hand_time[i]:
                count_state_arm += 1
            synced_idx3.append(count_state_arm)

        print("# sync: ", len(state_hand_time), len(synced_idx0), len(synced_idx1), len(synced_idx2), len(synced_idx3))

        for i in range(5, len(state_hand_time) - 5):
            pkl_data['state_hand'].append(self.state_hand_data[i])
            pkl_data['image'].append(self.image_data[synced_idx0[i]])
            pkl_data['cmd_hand'].append(self.cmd_hand_data[synced_idx1[i]])
            pkl_data['cmd_xarm'].append(self.cmd_xarm_data[synced_idx2[i]])
            pkl_data['state_xarm'].append(self.state_xarm_data[synced_idx3[i]])

        print("# cmd_xarm data: ", len(pkl_data['cmd_xarm']))
        print("# cmd_hand data: ", len(pkl_data['cmd_hand']))
        print("# image data: ", len(pkl_data['image']))
        print("# state_hand data: ", len(pkl_data['state_hand']))
        print("# state_xarm data: ", len(pkl_data['state_xarm']))

        with open(data_filename, 'wb') as f:
            pkl.dump(pkl_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_file_name', '-f', default='')
    parser.add_argument('--load_file_path', '-p', default='')
    parser.add_argument('--save', '-s', action='store_true')
    args = parser.parse_args()

    path_to_data = args.load_file_path
    file_time = args.load_file_name

    bag_path = osp.join(path_to_data, file_time, "rosbags/")
    save_path = osp.join(path_to_data, file_time, "pkls/")
    files = os.listdir(bag_path)

    for file in files:
        print("File name: ", file)
        reader = RosbagReader(bag_path, file, save_path, file_time)
        reader.read_data()
        if args.save:
            reader.save_pkl()
