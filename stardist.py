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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will hide INFO and WARNING messages.

# Define the path to the directory containing the .tif images
path = r"/content/drive/MyDrive/Colab_Notebooks/tapis_cellulaire_WT/"

# Change the working directory to the path where the images are located
os.chdir(path)

# This list holds all the image filenames
whole_image = []

# Creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # Loops through each file in the directory
    for file in files:
        if file.name.endswith('.tif'):
            # Adds only the image files to the nuclei list
            whole_image.append(file.name)

# Create the destination directories if they do not exist
path_Channel_2 = "/content/drive/MyDrive/Colab_Notebooks/PCNA/"
path_Channel_1 = "/content/drive/MyDrive/Colab_Notebooks/HOECHST/"
path_masks = "/content/drive/MyDrive/Colab_Notebooks/Mask/"

if not os.path.exists(path_Channel_2):
    os.makedirs(path_Channel_2)

if not os.path.exists(path_Channel_1):
    os.makedirs(path_Channel_1)

if not os.path.exists(path_masks):
    os.makedirs(path_masks)



# Iterate over each image in the nuclei list
for image_name in whole_image:
    # Set the path of the original image to a variable and target path
    image_path = os.path.join(path, image_name)

    # Load the two-channel image stack
    image_stack = skimage.io.imread(image_path)

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.imshow(image_stack[0, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_1")
    plt.subplot(1,2,2)
    plt.imshow(image_stack[1, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_2")

     # Select the first channel 0 or second channel 1 for nuclei segmentation
    image_full = image_stack[0, :, :]
    image = rescale(image_full, 0.33, anti_aliasing=False)
    #image=transform.rescale(image_full, 1/3, anti_aliasing=False, multichannel=False, mode='reflect')

    # Load a pre-trained StarDist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')


    # Predict instance segmentation on the example image
    labels, details = model.predict_instances(normalize(image))

    def remove_border_objects(labels):
      # Create a binary mask of the labels
      mask = labels > 0

      # Remove objects touching the border of the image
      border = np.zeros_like(mask)
      border[0, :] = True
      border[-1, :] = True
      border[:, 0] = True
      border[:, -1] = True
      mask &= ~border

      # Update the labels to exclude objects touching the border
      labels[~mask] = 0

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

    # Iterate over each label to save the nuclei as independent images
    for label in unique_labels[1:]:
        mask = labels == label
        nucleus = np.zeros_like(image_stack)
        nucleus[0, :, :] = image_stack[0, :, :] * mask
        nucleus[1, :, :] = image_stack[1, :, :] * mask
        # Crop the nucleus by finding the bounds of the nucleus in the mask
        coords = np.column_stack(np.where(mask))
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        crop = nucleus[:, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        file_name = "{}_nucleus_{}_Channel_1.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_1, file_name)
        skimage.io.imsave(full_path, crop[0, :, :].astype(np.uint16))
        file_name = "{}_nucleus_{}_Channel_2.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_2, file_name)
        skimage.io.imsave(full_path, crop[1, :, :].astype(np.uint16))

    # Remove objects touching the border of the image
    labels = remove_border_objects(labels)
    # Save the image displayed by plt.imshow to a file object
    fig = plt.imshow(render_label(labels, img=image_full))
    plt.axis("off")
    plt.savefig("output.tif", format='tif')
    plt.gcf().clear()
    plt.close()

    # Read the image data from the file object
    image_data = skimage.io.imread("output.tif")
    full_path = os.path.join(path_masks, "{}_mask.tif".format(base_name))
    # Save the image data as TIF
    skimage.io.imsave(full_path, image_data)

    # Save the image displayed by plt.imshow to a file object
    fig = plt.imshow(render_label(labels, img=image_full))
    plt.axis("off")
    plt.savefig("output_border_removed.tif", format='tif')
    plt.gcf().clear()
