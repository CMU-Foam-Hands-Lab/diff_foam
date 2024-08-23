import time
from builtin_interfaces.msg import Time
import cv2
import numpy as np
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
        self.min_pos = np.array([0.000] * 23)
        self.max_pos = np.array([1.000] * 23)
        self.min_pos[3] = -1.000  
        self.max_pos[3] = 1.000  
        self.min_pos[11] = -1.000 
        self.max_pos[11] = 1.000 
        self.min_pos[16:] = np.array([-0.28, -0.78, -1.19, 0.13, -0.15, 0.14, -2.79])
        self.max_pos[16:] = np.array([0.66,  0.20,  0.17,  1.67, 1.06,  1.68, -0.71])

        self.init_hand_pos = np.array([0.0, 0.0, 0.0285, -0.0295, 0.3258, 0.0, 0.0, 0.0, 0.1020, 0.0, 0.0, -0.0707, 0.0701, 0.0274, 0.0, 0.0143]) 
        self.init_xarm_pos = np.array([-0.0199, -0.1503, 0.0307, 1.3668, 0.0905, 1.4956, -1.7334])
        self.init_pos = np.concatenate([self.init_hand_pos, self.init_xarm_pos])
        self.image = None

        if self.enable_foam:
            self.autohand_pub = self.create_publisher(JointState, '/autohand_node/cmd_autohand', 100)
            self.autohand_sub = self.create_subscription(JointState, '/autohand_node/state_autohand', self.autohand_callback, 100)
            self.xarm_pub = self.create_publisher(JointState, '/diff/xarm/move_joint_cmd', 100)
            self.xarm_sub = self.create_subscription(JointState, '/xarm/joint_states', self.xarm_callback, 100)
            time.sleep(1.0)

            if self.reset_foam:
                self.reset()
                time.sleep(1.0)

        if self.enable_camera:
            self.br = CvBridge()
            self.img_sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.img_callback, 100)
        self.get_logger().info("Initialized Foamhand and Xarm!")
        
    def autohand_callback(self, joint_msg):
        # self.get_logger().info("Foamhand callback triggered.")
        self.init_hand_pos = np.array(joint_msg.position[:16])

    def xarm_callback(self, joint_msg):
        # self.get_logger().info("Xarm callback triggered.")
        self.init_xarm_pos = np.array(joint_msg.position[16:23])

    def img_callback(self, img):
        # self.get_logger().info("Camera image callback triggered.")
        np_arr = np.frombuffer(img.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def step(self, action):
        # self.get_logger().info("Step method called.")
        self.timestamp = Time()
        self.timestamp.sec = int(self.get_clock().now().nanoseconds / 1e9)
        self.timestamp.nanosec = self.get_clock().now().nanoseconds % int(1e9)

        assert len(action) == self.num_motors, "Action length must match the number of motors"

        action = np.clip(action, self.min_pos, self.max_pos).tolist()
        # action = action.tolist()


        hand_action = action[:16]  # First 16 dimensions for FoamHand
        xarm_action = action[16:]  # Last 7 dimensions for XArm

        hand_control_msg = JointState()
        hand_control_msg.header.stamp = self.timestamp
        hand_control_msg.position = hand_action
        self.autohand_pub.publish(hand_control_msg)
        # self.get_logger().info(f"Hand control message position: {hand_control_msg.position}")
        # self.get_logger().info(f"Hand control message stamp: {hand_control_msg.header.stamp}")

        xarm_control_msg = JointState()
        xarm_control_msg.header.stamp = self.timestamp
        xarm_control_msg.position = xarm_action
        self.xarm_pub.publish(xarm_control_msg)
        # self.get_logger().info(f"XArm control message position: {xarm_control_msg.position}")
        # self.get_logger().info(f"XArm control message stamp: {hand_control_msg.header.stamp}")
        
        if self.enable_camera:
            img, oimg = self.get_image()
            # if img is None:
            #     self.get_logger().warn("No image received in step method.")
            # else:
            #     self.get_logger().info("Image received in step method.")
        else:
            img, oimg = None, None

        return {'image': img, 'agent_pos': action, 'oimage': oimg }

    def get_image(self):
        if self.image is None:
            return None, None
        cur_image = self.image.copy()
        viz_image = self.image.copy()
        res_img = cv2.resize(cur_image, dsize=(320, 240))
        return res_img, viz_image

    def reset(self):
        self.get_logger().info("Reset method called.")

        self.timestamp = Time()
        self.timestamp.sec = int(self.get_clock().now().nanoseconds / 1e9)
        self.timestamp.nanosec = self.get_clock().now().nanoseconds % int(1e9)
        
        hand_control_msg = JointState()
        hand_control_msg.header.stamp = self.timestamp
        hand_control_msg.position = self.init_hand_pos.tolist()
        # self.get_logger().info(f"Hand reset message position: {hand_control_msg.position}")
        # self.get_logger().info(f"Hand reset message stamp: {hand_control_msg.header.stamp}")
        self.autohand_pub.publish(hand_control_msg)
        
        xarm_control_msg = JointState()
        xarm_control_msg.header.stamp = self.timestamp
        xarm_control_msg.position = self.init_xarm_pos.tolist()
        # self.get_logger().info(f"XArm reset message position: {xarm_control_msg.position}")
        # self.get_logger().info(f"XArm reset message stamp: {xarm_control_msg.header.stamp}")
        self.xarm_pub.publish(xarm_control_msg)
        time.sleep(0.1)
