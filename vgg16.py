import numpy as np
import os
from keras.applications.vgg16 import VGG16, preprocess_input
import imageio
import pandas as pd
import cv2
import csv

# Chemin de l'image d'entrée
input_img = '/content/gdrive/MyDrive/Colab_Notebooks/PCNA_petit/Copie_de_240222_drug_05_nucleus_182.tif'
img = imageio.imread(input_img)

# Convertir l'image en BGR (OpenCV utilise BGR au lieu de RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Redimensionner l'image à 224x224
img = cv2.resize(img, (224, 224))

# Ajouter une dimension pour le nombre d'échantillons d'entrée
img = np.expand_dims(img, axis=0)

# Définir les dimensions pour l'entrée du modèle
input_shape = (224, 224, 3)

# Modèle VGG16 avec les poids pré-entraînés
model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Prétraiter les données pour qu'elles puissent être alimentées au modèle VGG16
prep_input_vgg16 = preprocess_input(img)

# Extract features from the first 5 layers
layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1']
vgg16_features = []
for name in layer_names:
    layer = [l for l in model_vgg16.layers if l.name == name][0]
    feature = model_vgg16.predict(prep_input_vgg16)[0]
    vgg16_features.append(feature[:, :, layer.output_shape[-1]])

# Print shapes of the extracted features
for i, feature in enumerate(vgg16_features):
    print(f'Feature shape for layer {layer_names[i]}:', feature.shape)

# Save features to a CSV file
output_folder = '/content/gdrive/MyDrive/Colab_Notebooks/PCNA_petit/blabla/'
filename = os.path.join(output_folder, 'vgg16_features.csv')

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Layer', 'Feature'])
    for i, feature in enumerate(vgg16_features):
        feature_array = feature.flatten()
        writer.writerow([layer_names[i]] + list(feature_array))

# Read features from the CSV file
features = pd.read_csv(filename)

# Print the number of rows and columns in the features DataFrame
print('Number of rows:', len(features))
print('Number of columns:', len(features.columns))

# Print the first 5 rows of the features DataFrame
print(features.head())

