import os  
import sys 
import argparse  
import subprocess 
import pickle as pkl 
import collections 
import math 
import time 

import cv2 
import numpy as np 
import taichi as ti 
import torch 
import torch.nn as nn 
import torchvision 
import rclpy 
from rclpy.node import Node 

from typing import Tuple, Sequence, Dict, Union, Optional, Callable 

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler 
from diffusers.training_utils import EMAModel 
from diffusers.optimization import get_scheduler 

from autohand_node import AutoHandNode 
from dataset.data_utils import * 
from model.noise_pred_net import * 
from model.visual_encoder import * 
from config import * 
from foam_env import *

# Define a function to save ROS bag
def save_rosbag(dir_path: str, filename: str = "", topic: str = '/autohand_node/cmd_autohand /xarm/move_joint_cmd /autohand_node/state_autohand /xarm/joint_states /camera/color/image_raw/compressed') -> subprocess.Popen:
    try:
        cmd = ["ros2", "bag", "record", "-o", os.path.join(dir_path, filename), topic]  # Form the command to record ROS bag
        return subprocess.Popen(cmd, stdout=subprocess.PIPE)  # Execute the command and return the subprocess object
    except Exception as e:
        print(f"Failed to start ROS bag recording: {e}")  # Print error message if an exception occurs
        raise

# Define the DiffPolicyInfer class
class DiffPolicyInfer:
    def __init__(self, file_name: str, save_flag: bool, record_flag: bool, ckpt_path: str, save_path: str):
        try:
            rclpy.init()  # Initialize ROS 2 client library
            self.node = rclpy.create_node("foam_hand_auto")  # Create a ROS 2 node

            self.num_motors = 23  # Set the number of motors
            self.min_pos = [-1] * self.num_motors  # Set the minimum position of the motors
            self.max_pos = [1] * self.num_motors  # Set the maximum position of the motors

            # Get and concatenate the initial values of hand positions
            init_hand_pos_port1 = AutoHandNode.init_pos[0]
            init_hand_pos_port2 = AutoHandNode.init_pos[1]
            self.ini_hand_pos = np.concatenate((init_hand_pos_port1, init_hand_pos_port2), axis=1)
            self.ini_xarm_pos = np.array([0, 0, 0, 1.57, 0, 1.57, 0])  # Set the initial position of the XArm
            self.joint_positions = np.concatenate([self.ini_hand_pos, self.ini_xarm_pos])  # Concatenate hand and XArm initial positions

            # Initialize visual encoder and noise prediction network
            vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))
            noise_pred_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon)
            self.nets = nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})

            # Initialize noise scheduler
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=80,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

            self.device = torch.device('cuda')  # Set the device to GPU
            self.nets.to(self.device)  # Move networks to GPU

            self.ckpt_path = os.path.join(ckpt_path, file_name + ".pt")  # Set the checkpoint path
            if not os.path.isfile(self.ckpt_path):  # Check if the checkpoint file exists
                raise FileNotFoundError("Checkpoint file not found!")

            state_dict = torch.load(self.ckpt_path, map_location='cuda')  # Load the checkpoint
            self.ema_nets = self.nets
            self.ema_nets.load_state_dict(state_dict['model_state_dict'])  # Load the model state dictionary
            print('Pretrained weights loaded.')  # Print a message indicating that pretrained weights are loaded

            self.max_steps = 200  # Set the maximum number of steps
            self.cur_img = None  # Initialize the current image
            self.cur_action = None  # Initialize the current action

            self.foam_env = FoamEnv(enable_foam=True, enable_camera=True, reset_foam=True)  # Initialize the Foam environment

            init_action = self.joint_positions  # Set the initial action
            init_obs = self.foam_env.step(init_action)  # Execute the action and get the initial observation
            self.cur_img = init_obs['oimage']  # Get the current image
            self.cur_action = init_action  # Set the current action

            self.obs_deque = collections.deque([init_obs] * obs_horizon, maxlen=obs_horizon)  # Initialize the observation queue

            self.save_flag = save_flag  # Set the save flag
            self.record_flag = record_flag  # Set the record flag
            if self.save_flag or self.record_flag:  # If saving or recording is required
                self.save_file_name = input("Enter the file name to save data: ")  # Get the file name to save data
                self.save_path = save_path  # Set the save path

            self.in_record = False  # Initialize recording state
            if self.record_flag:  # If recording is required
                self.start_saving(self.save_file_name)  # Start saving data
        except Exception as e:
            print(f"Initialization failed: {e}")  # Print error message if initialization fails
            rclpy.shutdown()  # Shut down ROS 2 client library
            raise

    def finish(self):
        try:
            if self.save_flag or self.record_flag:  # If saving or recording is required
                self.end_saving()  # End saving data
            self.foam_env.reset()  # Reset the Foam environment
            time.sleep(1.0)  # Wait for 1 second
        except Exception as e:
            print(f"Error during finish: {e}")  # Print error message if an error occurs during finish
        finally:
            rclpy.shutdown()  # Shut down ROS 2 client library

    def start_saving(self, file_name: str):
        try:
            if not self.in_record:  # If not currently recording
                self.rosbag_p = save_rosbag(self.save_path, file_name + ".bag")  # Start ROS bag recording
                self.in_record = True  # Update recording state
        except Exception as e:
            print(f"Failed to start saving: {e}")  # Print error message if starting saving fails

    def end_saving(self):
        try:
            if self.in_record:  # If currently recording
                self.rosbag_p.terminate()  # Terminate ROS bag recording
                self.in_record = False  # Update recording state
        except Exception as e:
            print(f"Failed to end saving: {e}")  # Print error message if ending saving fails

    def step(self) -> np.ndarray:
        try:
            images = np.stack([x['image'] for x in self.obs_deque])  # Stack images into an array
            agent_poses = np.stack([x['agent_pos'] for x in self.obs_deque])  # Stack agent poses into an array

            nagent_poses = normalize_data(agent_poses, stats=stats)  # Normalize agent poses
            images = np.moveaxis(images, -1, 1)  # Adjust axes of the image array
            nimages = normalize_images(images)  # Normalize images

            nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)  # Convert images to PyTorch tensor and move to device
            nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)  # Convert agent poses to PyTorch tensor and move to device

            with torch.no_grad():  # In a context where gradients are not calculated
                image_features = self.ema_nets['vision_encoder'](nimages)  # Get image features
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)  # Concatenate image features and agent poses
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)  # Process observation condition

                noisy_action = torch.randn((1, pred_horizon, action_dim), device=self.device)  # Generate noisy action
                naction = noisy_action  # Initialize action

                self.noise_scheduler.set_timesteps(80)  # Set the number of timesteps for the noise scheduler

                for k in self.noise_scheduler.timesteps:  # For each timestep
                    noise_pred = self.ema_nets['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)  # Predict noise
                    naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample  # Update action

            naction = naction.detach().cpu().numpy()[0]  # Move action from GPU to CPU and convert to numpy array
            action_pred = unnormalize_data(naction, stats=stats)  # Unnormalize action

            start = obs_horizon - pred_horizon  # Calculate start index
            end = obs_horizon  # Calculate end index

            self.cur_action = action_pred[start:end]  # Update current action

            obs = self.foam_env.step(self.cur_action[0])  # Execute the action and get the observation
            self.obs_deque.append(obs)  # Append the observation to the deque

            self.cur_img = obs['image']  # Update current image

            return self.cur_img  # Return the current image
        except Exception as e:
            print(f"Error during step: {e}")  # Print error message if an error occurs during step
            raise

    def reset(self):
        try:
            self.foam_env.reset()  # Reset the Foam environment

            obs = self.foam_env.step(self.ini_hand_pos)  # Execute the initial action and get the observation
            self.cur_img = obs['image']  # Update current image

            self.obs_deque.clear()  # Clear the observation deque
            self.obs_deque.extend([obs] * obs_horizon)  # Fill the deque with initial observations

            return self.cur_img  # Return the current image
        except Exception as e:
            print(f"Error during reset: {e}")  # Print error message if an error occurs during reset
            raise

    def get_image(self):
        try:
            if self.cur_img is None:  # If the current image is not available
                raise ValueError("Current image is empty!")  # Raise an error
            return self.cur_img  # Return the current image
        except Exception as e:
            print(f"Error during get_image: {e}")  # Print error message if an error occurs during get_image
            raise

    def save_data(self, file_path: str, data: dict):
        try:
            with open(file_path, 'wb') as f:  # Open the file for writing in binary mode
                pkl.dump(data, f)  # Save the data to the file
        except Exception as e:
            print(f"Failed to save data: {e}")  # Print error message if saving data fails
            raise

def main(args=None):
    try:
        parser = argparse.ArgumentParser()  # Initialize argument parser

        # Add arguments
        parser.add_argument("--file_name", type=str, required=True, help="Name of the experiment.")
        parser.add_argument("--save_flag", action="store_true", help="Flag to save data.")
        parser.add_argument("--record_flag", action="store_true", help="Flag to record data.")
        parser.add_argument("--ckpt_path", type=str, default="", help="Path to the checkpoint file.")
        parser.add_argument("--save_path", type=str, default="", help="Path to save the recorded data.")

        args = parser.parse_args(args)  # Parse the arguments

        experiment = DiffPolicyInfer(
            file_name=args.file_name,
            save_flag=args.save_flag,
            record_flag=args.record_flag,
            ckpt_path=args.ckpt_path,
            save_path=args.save_path
        )  # Create an instance of DiffPolicyInfer class

        for step in range(experiment.max_steps):  # Loop through steps
            img = experiment.step()  # Execute a step
            cv2.imshow('Current Image', img)  # Display the current image
            if cv2.waitKey(1) == ord('q'):  # If 'q' is pressed, break the loop
                break

        experiment.finish()  # Finish the experiment
        cv2.destroyAllWindows()  # Destroy all OpenCV windows
    except Exception as e:
        print(f"Error in main: {e}")  # Print error message if an error occurs in main

if __name__ == '__main__':
    main()  # Execute main function
