import cv2
from skimage.measure import regionprops, label
import os

def extract_features(image_path_pcna, image_path_hoechst):
    # Load images
    image_pcna = cv2.imread(image_path_pcna)
    image_hoechst = cv2.imread(image_path_hoechst)

    # Convert images to grayscale
    gray_image_pcna = cv2.cvtColor(image_pcna, cv2.COLOR_BGR2GRAY)
    gray_image_hoechst = cv2.cvtColor(image_hoechst, cv2.COLOR_BGR2GRAY)

    # Segment cells for each treatment (PCNA and Hoechst)
    _, binary_image_pcna = cv2.threshold(gray_image_pcna, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, binary_image_hoechst = cv2.threshold(gray_image_hoechst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Label regions for each treatment
    labeled_image_pcna = label(binary_image_pcna)
    labeled_image_hoechst = label(binary_image_hoechst)

    # Extract regions and their properties for each treatment
    regions_pcna = regionprops(labeled_image_pcna, intensity_image=gray_image_pcna)
    regions_hoechst = regionprops(labeled_image_hoechst, intensity_image=gray_image_hoechst)

    # Variable lists:
    pcna_intensity = None
    cell_area = None
    pcna_count = None
    hoechst_intensity = None

    # Extract features for PCNA treatment
    for region in regions_pcna:
        # Get the average PCNA intensity
        pcna_intensity = region.mean_intensity

        # Get the cell area
        cell_area = region.area

        # Count PCNA spots
        pcna_count = len(region.coords)

    # Extract features for Hoechst treatment
    for region in regions_hoechst:
        # Get the average Hoechst intensity
        hoechst_intensity = region.mean_intensity

    return pcna_intensity, hoechst_intensity, cell_area, pcna_count

# Script usage
image_path_pcna = "/content/sample_data/PCNA_petit/"
image_path_hoechst = "/content/sample_data/HOECSHT_petit/"
sample_names = []

# Initialize lists to store features for each treatment
pcna_intensities = []
hoechst_intensities = []
cell_areas = []
pcna_counts = []

# Iterate through directories and process each image
for filename in os.listdir(image_path_hoechst):
    if filename.endswith(".tif"):  # You can adjust the extension based on your files
        try:
            sample_name = filename.split(".")[0]  # Get the sample name from the filename
            pcna_image_path = os.path.join(image_path_pcna, filename)
            hoechst_image_path = os.path.join(image_path_hoechst, filename)

            pcna_intensity, hoechst_intensity, cell_area, pcna_count = extract_features(pcna_image_path, hoechst_image_path)
            pcna_intensities.append(pcna_intensity)
            hoechst_intensities.append(hoechst_intensity)
            cell_areas.append(cell_area)
            pcna_counts.append(pcna_count)

            # Display extracted features for each cell for each treatment
            for i in range(len(sample_names)):
                print("Cell {} - PCNA Intensity: {}, Hoechst Intensity: {}, Area: {}, PCNA Count: {}".format(i+1, pcna_intensities[i], hoechst_intensities[i], cell_areas[i], pcna_counts[i]))
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

# Display extracted features for each cell for each treatment
for i in range(len(pcna_intensities)):
    if i < len(hoechst_intensities) and i < len(cell_areas) and i < len(pcna_counts):
        print("Cell {} - PCNA Intensity: {}, Hoechst Intensity: {}, Area: {}, PCNA Count: {}".format(i+1, pcna_intensities[i], hoechst_intensities[i], cell_areas[i], pcna_counts[i]))
    else:
        print("Number of cells extracted from PCNA and Hoechst images does not match.")

