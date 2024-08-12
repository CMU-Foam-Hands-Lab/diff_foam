#!/usr/bin/env python3

import numpy as np
import rclpy
import time
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from autohands_driver.dynamixel_client import DynamixelClient
import logging


# ROS node to control the hand

PORT1 = '/dev/ttyUSB1'
PORT2 = '/dev/ttyUSB2'
BAUD = 4000000
CALIBRATION = False
WRIST_CONTROL = False


class AutoHandNode(Node):
    def __init__(self):
        super().__init__("autohand_node")
        # get the parameters from the parameter server
        self.declare_parameter("motors", 'None')
        self.motor_params = self.get_parameter("motors").value

        # instantiate timer callbacks
        self.create_timer(0.2, self._publish_state)
        self.create_timer(0.2, self._write)
        self.create_timer(0.5, self._read)

        self.timestamp = Time()

        import yaml
        with open(self.motor_params, 'r') as file:
            motor_params = yaml.safe_load(
                file)['motors']
        self.motors_all = [motor_info["id"]
                           for motor_info in motor_params.values()]
        self.motor_names_all = [motor_info["name"]
                                for motor_info in motor_params.values()]
        self.upper_limits_all = [motor_info["limits"]["max"]
                                 for motor_info in motor_params.values()]
        self.lower_limits_all = [motor_info["limits"]["min"]
                                 for motor_info in motor_params.values()]
        self.kP_gains_all = [motor_info["gains"]["P"]
                             for motor_info in motor_params.values()]
        self.kI_gains_all = [motor_info["gains"]["I"]
                             for motor_info in motor_params.values()]
        self.kD_gains_all = [motor_info["gains"]["D"]
                             for motor_info in motor_params.values()]
        self.curr_limits_all = [motor_info["curr_limit"]
                                for motor_info in motor_params.values()]
        self.velocity_limits_all = [motor_info["vel_limit"]
                                    for motor_info in motor_params.values()]

        self.ports_all = [motor_info["port"]
                          for motor_info in motor_params.values()]

        self.motors = [[motor for motor, port in zip(self.motors_all,
                        self.ports_all) if port == PORT1],
                       [motor for motor, port in zip(self.motors_all, self.ports_all) if port == PORT2]]
        self.motor_names = [[name for name, port in zip(self.motor_names_all,
                                                        self.ports_all) if port == PORT1],
                            [name for name, port in zip(self.motor_names_all,
                                                        self.ports_all) if port == PORT2]]
        self.upper_limits = [[upper for upper, port in zip(self.upper_limits_all,
                             self.ports_all) if port == PORT1],
                             [upper for upper, port in zip(self.upper_limits_all, self.ports_all) if port == PORT2]]
        self.upper_limits_all = np.array(
            [self.upper_limits[0], self.upper_limits[1]]).reshape(1, -1).flatten()
        # flatten

        self.lower_limits = [[lower for lower, port in zip(self.lower_limits_all,
                                                           self.ports_all) if port == PORT1],
                             [lower for lower, port in zip(self.lower_limits_all, self.ports_all) if port == PORT2]]
        self.lower_limits_all = np.array(
            [self.lower_limits[0], self.lower_limits[1]]).reshape(1, -1).flatten()
        self.kP_gains = [[kP for kP, port in zip(self.kP_gains_all,
                                                 self.ports_all) if port == PORT1],
                         [kP for kP, port in zip(self.kP_gains_all, self.ports_all) if port == PORT2]]
        self.kI_gains = [[kI for kI, port in zip(self.kI_gains_all,
                                                 self.ports_all) if port == PORT1],
                         [kI for kI, port in zip(self.kI_gains_all, self.ports_all) if port == PORT2]]
        self.kD_gains = [[kD for kD, port in zip(self.kD_gains_all,
                                                 self.ports_all) if port == PORT1],
                         [kD for kD, port in zip(self.kD_gains_all, self.ports_all) if port == PORT2]]
        self.curr_limits = [[curr for curr, port in zip(self.curr_limits_all,
                            self.ports_all) if port == PORT1],
                            [curr for curr, port in zip(self.curr_limits_all, self.ports_all) if port == PORT2]]
        self.velocity_limits = [[vel_lim for vel_lim, port in zip(self.velocity_limits_all,
                                self.ports_all) if port == PORT1],
                                [vel_lim for vel_lim, port in zip(self.velocity_limits_all, self.ports_all) if port == PORT2]]
        self.ports = [PORT1, PORT2]
 
        for motor_name, motor_info in motor_params.items():
            motor_id = motor_info["id"]
            # Access other parameters specific to each motor
            motor_name = motor_info["name"]

            print(f"Motor Name: {motor_name}, ID: {motor_id}")

        self.curr_pos = np.zeros([2, len(self.motors[0])])
        self.prev_pos = np.zeros([2, len(self.motors[0])])
        self.goal_pos = np.zeros([2, len(self.motors[0])])
        self.init_pos = np.zeros([2, len(self.motors[0])])

        # subscribe to hand commands, which are the desired motor positions in radians
        _ = self.create_subscription(JointState, "/autohand_node/cmd_autohand", self._update_goal_pos, 10)

        # publisher for the hand state
        self.pub = self.create_publisher(JointState, "/autohand_node/state_autohand", 10)

        # get the motor IDs from the cfg file

        # # connect to the motors and set the default parameters
        # ##########################################################
        # ##########################################################
        # HOMING ROUTINE
        # ##########################################################
        # ##########################################################
        self.dual_tendon = np.zeros(
            [len(self.motors[0]) + len(self.motors[1])])
        
        logging.info("Starting homing routine")

        for idx_port in [0, 1]:
            # for idx_port in [0]:
            for name, motor, upper_lim, idx_motor in zip(self.motor_names[idx_port], self.motors[idx_port], self.upper_limits[idx_port], range(len(self.motors[idx_port]))):
                motor_id = [motor]
                dxl_client = DynamixelClient(
                    motor_id, self.ports[idx_port], BAUD)
                dxl_client.connect()
                self.get_logger().info("Connected to client!")
                curr_pos = dxl_client.read_sync_pos(retries=10)
                print(f"Motor ID: {motor}, Current Sync Position: {curr_pos}")

                dxl_client.sync_write(motor_id, [0], 20, 4)  # homing offset

                dxl_client.sync_write(
                    motor_id, np.ones(len(motor_id))*5, 11, 1)  # 5 mode
                dxl_client.sync_write(
                    motor_id, [500], 102, 2)  # current limit
                dxl_client.set_torque_enabled(motor_id, True, retries=2)
                dxl_client.sync_write(motor_id, [800], 84, 2)  # Pgain
                dxl_client.sync_write(motor_id, [0], 82, 2)  # Igain
                dxl_client.sync_write(motor_id, [200], 80, 2)  # Dgain

                # pull in tendon until current reaches threshold
                if upper_lim > 0:
                    dpos = 0.05
                else:
                    dpos = -0.05
                max_iter = 1000

                if CALIBRATION:
                    cur_max = 40
                    cur_min = -40
                else:
                    cur_max = 1
                    cur_min = -1

                # if motor == 41 or motor == 42:
                #     dxl_client.write_desired_pos(motor_id, np.array([0.0]))
                #     self.init_pos[idx_port][idx_motor] = 0.0
                #     dxl_client.disconnect()
                #     continue
                # if dual tendon, continue
                # if "ring_23" in name:
                #     print("here")
                if WRIST_CONTROL is False:
                    if "wrist" in name:
                        curr_pos = dxl_client.read_sync_pos(retries=10)
                        print(
                            f"Setting init pos: Motor ID: {motor}, Current Sync Position: {curr_pos}")
                        self.init_pos[idx_port][idx_motor] = curr_pos
                        dxl_client.disconnect()
                        continue

                if "dual" in name:
                    self.dual_tendon[idx_port *
                                     len(self.motors[0]) + idx_motor] = 1
                    # Dual tendon
                    max_pos = 0
                    min_pos = 0
                    # read current, increase until threshold, then do the same
                    # thing the other direction, then set init pos to
                    for i in range(max_iter):
                        curr_cur = dxl_client.read_cur()
                        curr_pos = dxl_client.read_sync_pos(retries=10)
                        if curr_cur > cur_max:
                            max_pos = curr_pos-dpos
                            break
                        dxl_client.write_desired_pos(motor_id, curr_pos + dpos)
                        time.sleep(0.1)
                    dpos = -dpos
                    for i in range(max_iter):
                        curr_cur = dxl_client.read_cur()
                        curr_pos = dxl_client.read_sync_pos(retries=10)
                        if curr_cur < cur_min:
                            min_pos = curr_pos-dpos
                            break
                        dxl_client.write_desired_pos(motor_id, curr_pos + dpos)
                        time.sleep(0.1)
                    # get init pos
                    self.init_pos[idx_port][idx_motor] = (max_pos+min_pos)/2
                    # self.init_pos[idx_port][idx_motor] = 0.0
                    print(
                        f"Setting dual tendon init pos: Motor ID: {motor}, Center Position: {self.init_pos[idx_port][idx_motor]}")
                    dxl_client.disconnect()
                    continue

                # all other tendons
                for i in range(max_iter):
                    curr_cur = dxl_client.read_cur()
                    # print(f"Motor ID: {motor}, Current Cur reading: {curr_cur}")
                    curr_pos = dxl_client.read_sync_pos(retries=10)
                    if curr_cur > cur_max or curr_cur < cur_min:
                        break
                    dxl_client.write_desired_pos(motor_id, curr_pos + dpos)
                    time.sleep(0.1)
                # reset init pos
                curr_pos = dxl_client.read_sync_pos(retries=10)
                print(
                    f"Setting init pos: Motor ID: {motor}, Current Sync Position: {curr_pos}")
                self.init_pos[idx_port][idx_motor] = curr_pos-3*dpos
                dxl_client.disconnect()

            for name, motor, idx_motor in zip(self.motor_names[idx_port], self.motors[idx_port], range(len(self.motors[idx_port]))):
                # if dual tendon, continue
                # if "dual" in name:
                #     continue
                motor_id = [motor]
                dxl_client = DynamixelClient(
                    motor_id, self.ports[idx_port], BAUD)
                dxl_client.connect()
                curr_pos = dxl_client.read_sync_pos(retries=10)
                print(
                    f"Verification: Motor ID: {motor}, Current Sync Position: {curr_pos}")
                print(
                    f"Verification: Motor ID: {motor}, Saved init Position: {self.init_pos[idx_port][idx_motor]}")
                dxl_client.write_desired_pos(
                    motor_id, np.array([self.init_pos[idx_port][idx_motor]]))
                time.sleep(0.1)
                dxl_client.disconnect()

        ##########################################################
        # finished homing routine. set up clients
        ##########################################################
        try:
            self.dxl_client = [DynamixelClient(
                self.motors[0], self.ports[0], BAUD),
                DynamixelClient(self.motors[1], self.ports[1], BAUD)]
            self.dxl_client[0].connect()
            self.dxl_client[1].connect()
        except Exception:
            print("Could not connect to the hand. Check the port and baudrate.")
        for idx_port in [0, 1]:
            # for idx_port in [0]:
            self.dxl_client[idx_port].set_torque_enabled(
                self.motors[idx_port], False, retries=2)
            self.dxl_client[idx_port].write_desired_pos(
                self.motors[idx_port], self.init_pos[idx_port])
            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], np.ones(len(self.motors[idx_port]))*5, 11, 1)  # 4 for extended position mode
            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.curr_limits[idx_port], 102, 2)  # current limit
            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.velocity_limits[idx_port], 108, 4)  # velocity limit
            # enable torque
            self.dxl_client[idx_port].set_torque_enabled(
                self.motors[idx_port], True, retries=2)
            self.dxl_client[idx_port].write_desired_pos(
                self.motors[idx_port], self.init_pos[idx_port])
            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.kP_gains[idx_port], 84, 2)  # Pgain

            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.kI_gains[idx_port], 82, 2)  # Igain
            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.kD_gains[idx_port], 80, 2)  # Dgain

            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.curr_limits[idx_port], 102, 2)  # current limit
            self.dxl_client[idx_port].sync_write(
                self.motors[idx_port], self.velocity_limits[idx_port], 108, 4)
        # set current goal_pos to
        self.goal_pos = np.copy(self.init_pos)
        print("Hand driver initialized")

    # callback to update the goal position of the hand whene a new command is received

    def _update_goal_pos(self, msg):
        position_cmd = np.zeros(len(self.motors_all))
        if len(msg.position) != len(self.motors_all):
            print("The number of goal positions does not match the number of motors")
            return
        for i, pos in enumerate(msg.position):
            # if i in any self.dual_tendon

            if self.dual_tendon[i] == 1:
                # clip between -1 and 1
                position_cmd[i] = np.clip(pos, -1, 1)
                position_cmd[i] = 0.5 * position_cmd[i] * \
                    (self.upper_limits_all[i]-self.lower_limits_all[i])
            else:
                # clip between 0 and 1
                position_cmd[i] = np.clip(pos, 0, 1)
                position_cmd[i] = self.lower_limits_all[i] + \
                    position_cmd[i]*(self.upper_limits_all[i] -
                                     self.lower_limits_all[i])
        # reshape into 2xN array
        position_cmd = np.reshape(position_cmd, [2, -1])
        self.goal_pos = self.init_pos + np.copy(position_cmd)
        # print("setting goal pos to " + str(self.goal_pos))

    # read the current position of the hand and write the new position at a certain rate
    def _write(self, event=None):
        # TODO write in parallel?
        for idx_port in [0, 1]:
            # for idx_port in [0]:
            self.dxl_client[idx_port].write_desired_pos(
                self.motors[idx_port], self.goal_pos[idx_port])

    def _read(self, event=None):
        for idx_port in [0, 1]:
            self.curr_pos[idx_port] = self.dxl_client[idx_port].read_sync_pos()

    # publish hand state at certain rate

    def _publish_state(self, event=None):
        state = JointState()
        self.timestamp.sec = int(self.get_clock().now().nanoseconds / 1e9)
        self.timestamp.nanosec = self.get_clock().now().nanoseconds % int(1e9)
        state.header.stamp = self.timestamp
        # state.name = [""]
        state.position = np.reshape(
            self.curr_pos-self.init_pos, [1, -1]).tolist()[0]
        self.pub.publish(state)

    def disconnect(self):
        for idx_port in [0, 1]:
            self.dxl_client[idx_port].disconnect()
        print("Hand driver disconnected")

# init the Autohand node


def main(args=None):
    rclpy.init(args=args)
    autohand_node = AutoHandNode()
    try:
        rclpy.spin(autohand_node)
    except KeyboardInterrupt:
        pass
    autohand_node.disconnect() 
    autohand_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
