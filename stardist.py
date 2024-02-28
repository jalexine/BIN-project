from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
from skimage import data, color
from skimage.transform import rescale, resize
import os
from PIL import Image
import skimage.transform

model = StarDist2D.from_pretrained('2D_versatile_fluo')  # Specify the pretrained model to be used

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide error messages

# Define the path to the directory containing the .tif images
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
img_dir = os.path.join(script_dir, "img")  # Assume images are in the 'img' subdirectory

# This list holds all the image filenames
whole_image = []

# Create a ScandirIterator aliased as files
with os.scandir(img_dir) as files:
    # Loop through each file in the directory
    for file in files:
        if file.name.endswith('.tif'):  # If the name ends with .tif, add it to the list whole_image
            # Adds only the image files to the nuclei list
            whole_image.append(file.name)

# Create the destination directories if they do not exist in the script's directory
path_Channel_2 = os.path.join(script_dir, "PCNA_WT")
path_Channel_1 = os.path.join(script_dir, "HOECHST_WT")
path_masks = os.path.join(script_dir, "Mask_WT")

if not os.path.exists(path_Channel_2):
    os.makedirs(path_Channel_2)

if not os.path.exists(path_Channel_1):
    os.makedirs(path_Channel_1)

if not os.path.exists(path_masks):
    os.makedirs(path_masks)

# Iterate over each image in the nuclei list
for image_name in whole_image:
    # Set the path of the original image to a variable and target path
    image_path = os.path.join(img_dir, image_name)  # Full path of the image, e.g., "script_dir/img/image_name.tif"

    # Load the two-channel image stack
    image_stack = skimage.io.imread(image_path)  # Load the 2 channels of the image

    plt.figure(figsize=(8, 8))  # Specify the figure size
    plt.subplot(1, 2, 1)  # Divide the figure into a grid of subplots (subplot) with 1 row and 2 columns, select the first subplot for the next plot
    plt.imshow(image_stack[0, :, :], cmap='gray')
    plt.axis("off")  # Turn off axis display
    plt.title("Channel_1")  # Add a title to the plot

    # Same process for the second image of the stack
    plt.subplot(1, 2, 2)
    plt.imshow(image_stack[1, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_2")

    # Select the first channel 0 or second channel 1 for nuclei segmentation
    image_full = image_stack[0, :, :]
    image = rescale(image_full, 0.33, anti_aliasing=False)  # Resize the image to 33% of its original size

    # Predict instance segmentation on the example image
    labels, details = model.predict_instances(normalize(image))

    def remove_border_objects(labels):
        # Get the shape of the labels image
        height, width = labels.shape

        # Create a new mask with the same shape as the labels image
        mask = np.ones((height, width), dtype=bool)

        # Set the border pixels to 0 in the mask
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False

        # Get the labels of the objects touching the border
        border_labels = set(labels[~mask])

        # Set the labels of the border objects to 0
        for label in border_labels:
            labels[labels == label] = 0

        # Renumber the labels
        labels = skimage.measure.label(labels, background=0)

        return labels

    # Remove objects touching the border of the image
    labels = remove_border_objects(labels)

    # Upsample the labels to the original size
    upsampled_labels = resize(labels, image_full.shape, order=0, anti_aliasing=False, preserve_range=True)
    labels = upsampled_labels

    # Get the list of unique labels
    unique_labels = np.unique(labels)

    # Get the base name of the image file
    base_name, _ = os.path.splitext(os.path.basename(image_path))

    # Iterate over each label, skipping background label 0
    for label in unique_labels[1:]:
        # Create a mask for the current label
        mask = labels == label

        # Get the coordinates of all pixels belonging to the current label
        y_coords, x_coords = np.where(mask)

        # Determine the minimum and maximum coordinates in y and x to create a bounding box around the object
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Create a bounding box with a padding of 10 pixels around the object
        box_y_min = max(y_min - 10, 0)
        box_y_max = min(y_max + 10, image_stack.shape[1] - 1)
        box_x_min = max(x_min - 10, 0)
        box_x_max = min(x_max + 10, image_stack.shape[2] - 1)

        # Crop channels 1 and 2 using the calculated bounding box
        channel_1_crop = image_stack[0, box_y_min:box_y_max + 1, box_x_min:box_x_max + 1]
        channel_2_crop = image_stack[1, box_y_min:box_y_max + 1, box_x_min:box_x_max + 1]

        # Generate a filename for each cropped channel based on the base name of the image file and the current label
        file_name = "{}_Drug_nucleus_{}_Channel_1.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_1, file_name)

        # Save the cropped channel 1 as a TIF image in the 'path_Channel_1' directory
        skimage.io.imsave(full_path, channel_1_crop.astype(np.uint16))

        # Generate a filename for each cropped channel based on the base name of the image file and the current label
        file_name = "{}_Drug_nucleus_{}_Channel_2.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_2
