import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set folder paths
folder_path = "/home/foamlab/nw/mydata/dice_single/viz/nosync/round4/xarm"
output_folder = "/home/foamlab/nw/plots/try1/xarm"
os.makedirs(output_folder, exist_ok=True)

# Get the names of all txt files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Process only the first 5 files
# file_list = file_list[:5]

# Initialize a list to store column data from all files
all_data = []

# Read data from all files
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data = np.loadtxt(file_path)
    all_data.append(data)

# Ensure all data arrays have the same number of columns
num_columns = all_data[0].shape[1]

# Set a more distinctive color palette
colors = plt.get_cmap('tab20').colors

# Create individual plots for each column of data
for col in range(num_columns):
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the figure size
    
    # Handle irregular arrays
    for idx, data in enumerate(all_data):
        if data.shape[1] > col:  # Ensure column index is within the data range
            ax.plot(data[:, col], linestyle='-', linewidth=2, color=colors[idx % len(colors)], label=f'File {idx + 1}')
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Joint States", fontsize=12)
    ax.set_title(f"Joint {col + 1}", fontsize=14)

    # Save the current figure
    plot_filename = f"joint_{col + 1}.png"
    plt.savefig(os.path.join(output_folder, plot_filename), bbox_inches='tight', pad_inches=0.1)
    plt.close()

print(f"Plots have been saved in {output_folder}.")
