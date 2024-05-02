# Paws and Pixels: Mobile Dog Breed Identification Using Modern Deep-Learning Image Techniques
Harry Williams - MSc Computer Science, The University of Bath
## Description
This repository documents the code and models developed in support of the submitted thesis.
## Contents
#### pawsandpixels
- Contains the source code used to build the mobile application
#### apk
- Contains three .apk files, which can be used to download the application to a smart phone. Each .apk relates to a different Android processor, so please check which is compatible.
#### tensorflow_datasets/stanford_dogs
- Contains the Stanford Dogs dataset, as loaded by the tfds library
#### Feature Extraction Code
- Contains twelves .py files to reproduce results from using feature extraction techniques on the twelve pre-trained models.
#### Feature Extraction Models
Contains twelve subdirectories for each model, with each one containing the following:
- The keras model saved in SavedModel format
- The .tflite model (converted from SavedModel format)
- Learning curves (accuracy+loss) from training
#### Fine-Tuning Code
- Contains two .py files to reproduce results from using data augmentation and fine-tuning techniques.
- This configuration of hyperparameters returns the best results, but adjust learning rate to reproduce other results shown in thesis
- Ensure to comment/uncomment appropriate portions of code to unfreeze different layers of the model
- EfficientNetV2S_warmedup.keras and EVA02Small_warmedup.keras are required for this code to run and are found in assets/. These models are simply trained with all layers frozen (with data augmentation) prior to fine-tuning
#### Fine-tuned Models
Contains two subdirectories for each model, with each one containing the following:
- The keras model saved in .keras format
- The .tflite model
#### Confusion Matrices
- Contains two .ipynb files showing how confusion matrices were constructed
- Contains two .png files, one for each full confusion matrix
#### GRAD-Cam Heatmaps
- Contains the .ipynb file used to construct heatmaps seen in thesis
#### assets
- Contains two .keras files used for the fine-tuning process
- Contains labels.txt, a text file for all 120 classes
- Contains test_images.zip, the twenty test images used for performance comparison between existing applications
- Contains Dataset_Analysis.ipynb, used to show class distributions and see example images
