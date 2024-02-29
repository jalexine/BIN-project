# Project README

## Automatic Cell Cycle Phase Determination using Fluorescence Imaging and Machine Learning

### Overview

This project aims to automatically determine cell cycle phases through the combination of fluorescence imaging and machine learning. The process involves capturing 2D cell images of the strain [SPECIFY STRAIN HERE] treated with Hoechst and PCNA using a confocal microscope [ADD MORE DETAILS HERE].

### Implementation Steps

1. **Image Capture:**
   - Acquire 2D cell images using a confocal microscope, treating the specified strain with Hoechst and PCNA.

2. **Image Processing:**
   - Use Fiji to read images.
   - Apply StarDist 2D (using provided scripts) to generate masks and separate cells into PCNA and Hoechst channels.

3. **Feature Extraction:**
   - Utilize the VGG16 model to extract features from images.

4. **Statistical Analysis:**
   - Perform statistical analyses using extracted features for cell classification (PCA, t-SNE, K-means, DBSCAN).

5. **Results and Visualization:**
   - Analyze and visualize results, obtaining images of initial states for each identified cell cycle stage.

### Dependencies

Ensure the following dependencies are installed:

- [Fiji](https://imagej.net/Fiji/Downloads)
- [StarDist 2D](https://github.com/stardist/stardist) (Scripts provided in the repository)
- [VGG16 Model](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/jalexine/BIN-project.git
   cd your-repo
   ```
2. Follow the steps outlined in the "Implementation" section.
3. Review the results and visualizations obtained.

### Contributors

- Alexine LP
- Mael Keravis
```

