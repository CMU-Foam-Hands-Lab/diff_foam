# Diffusion Policy Deployment for Foam Hand and XArm7

Welcome to the Diffusion Policy project for deploying on Foam Hand and XArm. Below is a guide to help you get started with running this project.

## Project Structure

- `/diffusion_policy/.venv`  
  Virtual environment for running the project.

- `/mydata`  
  Contains collected data.

- `/datasets`  
  Contains preprocessing scripts to convert data from bag format to pkl format.

- `/echos`  
  Contains output data and analysis scripts from past tests.

- `/model`  
  Contains the visual encoder and noise prediction network for the diffusion model.

- `/plots`  
  Contains scripts for visualizing model outputs.

- `/save`  
  Directory for saving trained model weights.

## Training the Model

To train the model:

1. Run the `training_object.ipynb` notebook.
2. Make sure to update the data input and output paths in the notebook (the `nw` folder has been renamed to `diffusion_policy`).

## Inference

To run model inference:

1. Launch the RealSense camera node:
   ```bash
   ros2 launch realsense_rgbd_transport_ros2 rs_launch.py
   ```
   This should be run from the `~/rs_camera` directory.

2. Run the `inference.ipynb` notebook.
3. Ensure to update the data input and output paths in the notebook. 

Inference output data will be saved automatically to the `save/` folder. You can adjust the `save_flag` and `record_flag` in the `inference.ipynb` notebook to control whether the ROS bag is saved.

## Additional Information

For more details on the project, please refer to the poster available on Google Drive: https://drive.google.com/file/d/1hYUVsxyhiXEtXabWOoytsa8HlRqqGbKX/view?usp=drive_link.
