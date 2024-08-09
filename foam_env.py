import time
import cv2
import numpy as np
from autohand_node import AutoHandNode
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class FoamEnv(Node):
    def __init__(self, enable_foam, enable_camera, reset_foam):
        super().__init__('foam_env_node')
        
        self.enable_foam = enable_foam
        self.enable_camera = enable_camera
        self.reset_foam = reset_foam

        self.num_motors = 23
        self.min_pos = [-1] * self.num_motors 
        self.max_pos = [1] * self.num_motors  
        init_hand_pos_port1 = AutoHandNode.init_pos[0]
        init_hand_pos_port2 = AutoHandNode.init_pos[1]
        self.ini_hand_pos = np.concatenate((init_hand_pos_port1, init_hand_pos_port2), axis=1)  
        self.ini_xarm_pos = np.array([0, 0, 0, 1.57, 0, 1.57, 0])
        self.ini_pos = np.concatenate([self.ini_hand_pos, self.ini_xarm_pos])
        self.image = None

        if self.enable_foam:
            self.autohand_pub = self.create_publisher(JointState, '/autohand_node/cmd_autohand', 10)
            self.autohand_sub = self.create_subscription(JointState, '/autohand_node/state_autohand', self.autohand_callback, 10)
            self.xarm_pub = self.create_publisher(JointState, '/xarm/move_joint_cmd', 10)
            self.xarm_sub = self.create_subscription(JointState, '/xarm/joint_states', self.xarm_callback, 10)
            self.get_logger().info("Foamhand and Xarm joint state subscribed.")
            time.sleep(1.0)

            if self.reset_foam:
                self.reset()
                time.sleep(1.0)

        if self.enable_camera:
            self.br = CvBridge()
            self.img_sub = self.create_subscription(CompressedImage, "/camera/depth/image_rect_raw/compressed", self.img_callback, 10)
            self.get_logger().info("Camera subscribed.")

        time.sleep(1)
        self.get_logger().info("Initialized Foamhand and Xarm!")

    def autohand_callback(self, joint_msg):
        self.ini_hand_pos = np.array(joint_msg.position[:16])

    def xarm_callback(self, joint_msg):
        self.ini_xarm_pos = np.array(joint_msg.position[16:23])

    def img_callback(self, img):
        np_arr = np.frombuffer(img.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def step(self, action):
        action = np.clip(action, self.min_pos, self.max_pos).tolist()
        control_msg = JointState()
        assert len(action) == self.num_motors
        control_msg.position = action
        self.autohand_pub.publish(control_msg)  
        if self.enable_camera:
            img = self.get_image()
        else:
            img = None, None

        return {'image': img, 'agent_pos': action}

    def get_image(self):
        if self.image is None:
            return None, None
        cur_image = self.image.copy()
        viz_image = self.image.copy()
        res_img = cv2.resize(cur_image, dsize=(320, 240))
        return res_img, viz_image

    def reset(self):
        control_msg = JointState()
        control_msg.position = self.ini_pos.tolist()  
        self.autohand_pub.publish(control_msg)  
        time.sleep(0.1)
