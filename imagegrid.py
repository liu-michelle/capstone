import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Path to the folder containing images (same directory)
image_folder = './pics'

# Get a list of all image files in the folder (filter out non-image files if necessary)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Number of images
num_images = len(image_files)

# Calculate grid dimensions: try to get a square-ish grid
n_cols = int(np.ceil(np.sqrt(num_images)))  # Number of columns
n_rows = int(np.ceil(num_images / n_cols))  # Number of rows

# Create a figure with enough subplots to hold all images
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

# Flatten axes array in case of multi-row/column arrangement
axes = axes.flatten()

# Loop through the images and add them to the grid
for i, image_file in enumerate(image_files):
    # Load the image
    img_path = os.path.join(image_folder, image_file)
    img = mpimg.imread(img_path)
    
    # Display the image on the appropriate subplot
    axes[i].imshow(img)
    axes[i].axis('off')  # Hide axes

# Hide any remaining empty subplots
for i in range(num_images, len(axes)):
    axes[i].axis('off')

# Show the plot
plt.tight_layout()
plt.show()
