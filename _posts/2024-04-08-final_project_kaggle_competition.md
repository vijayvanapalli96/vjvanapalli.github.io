

## Kaggle Competition - Image Matching 2024
The goal of this competition is to create detailed 3D maps from collections of images across various settings and conditions. 

The primary metric used to evaluate submissions is the mean Average Accuracy (mAA) based on the positioning of registered camera centers. The registration of each camera is calculated relative to its "ground truth" position. 

According to the Kaggle competition rules, a camera is considered successfully registered if the distance between the transformed camera center T(C) and the ground truth camera center, is less than a predefined threshold t. This distance measures how well the predicted camera pose matches the actual pose. 

Calculation of mAA: 
For each scene the mAA is computed by averaging the registration accuracy rates( ri) for multiple thresholds t, allowing flexibility in precision based on the context of the scene. 

The way I approached this Kaggle competiton is by understanding the steps taken by a baseline Kaggle notebook and understanding each segment. 
## The Baseline Notebook breakdown:
## Embedding Images 
Here in this section of the notebook, the embed_images function is used to compute embeddings for a list of image paths using a specified model. Here the model is a Hugging Face's Transformers loaded via AutoImageProcessor and AutoModel. 
## Identifying Image Pairs 
The function get_image_pairs is designed to find pairs of similar images based on the computed embeddings. The approach includes several steps and parameters. 
Here the distance metric is used to calculate distances between all pairs of embeddings using 'torch.cdist'.  Here only pairs with a distance less than the specified 'similarity_threshold' are initially considered as similar. 

We use pre-trained models to get effective embeddings to try and capture the content and style of images, making them suitable for tasks requiring similarity checks for clustering. 
