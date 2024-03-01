import os
import cv2
import numpy as np
from skimage.transform import resize
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from keras.applications.vgg16 import preprocess_input as keras_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image


# Paths to directories containing images
directory_hoechst = "/content/HOECHST_small"
directory_pcna = "/content/PCNA_small"

# Initialize the VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=True)

# Target size for resizing the images
target_size = (224, 224)

# Function to read and preprocess images
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features with the VGG16 model
def extract_features(image):
    features = vgg_model.predict(image)
    return features.flatten()

# Initialize the array to store features and sample names
all_features = []
sample_names = []

# Iterate through directories and process each image
for filename in os.listdir(directory_hoechst):
    if filename.endswith(".tif"):  # You can adjust the extension based on your files
        try:
            sample_name = filename.split(".")[0]  # Retrieve the sample name from the file name
            sample_names.append(sample_name)  # Add the sample name to the list

            # Read and preprocess HOECHST image
            image_hoechst = preprocess_image(os.path.join(directory_hoechst, filename))
            # Extract features from HOECHST image
            features_hoechst = extract_features(image_hoechst)

            # Read and preprocess PCNA image
            image_pcna = preprocess_image(os.path.join(directory_pcna, filename))
            # Extract features from PCNA image
            features_pcna = extract_features(image_pcna)

            # Concatenate features from both images
            combined_features = np.concatenate((features_hoechst, features_pcna))

            # Add features to the array
            all_features.append(combined_features)
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

# Convert the array to a numpy array
all_features = np.array(all_features)

# Initialize the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Select the top 20 features with SelectFromModel
sfm = SelectFromModel(rf, threshold=-np.inf, max_features=50)
sfm.fit(all_features, np.ones(all_features.shape[0]))  # Use dummy labels for feature selection
selected_features = sfm.transform(all_features)

# Convert the selected features to a pandas DataFrame with sample names
df = pd.DataFrame(selected_features, columns=[f"Feature_{i+1}" for i in range(selected_features.shape[1])])
df["Sample_Name"] = sample_names

# Specify the path and file name to save the CSV
csv_path = '/content/selected_features.csv'

# Save the pandas DataFrame to CSV format
df.to_csv(csv_path, index=False)

print("Processing completed")

