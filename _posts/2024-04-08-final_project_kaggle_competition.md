

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

## Detecting keypoints with ALIKED 
ALIKED is a computer vision technique for identifying key points and descriptors in images using a neural network that is designed to be computationally lightweight. Its main innovation is the use of deformable transformations, allowing it to adapt to geometric changes in images, enhancing accuracy in feature extraction when compared to traditional methods like SIFT.

However this is exactly what I will be exploring later on! Using traditional methods like SIFT, ORB and AKAZE that are already available through the OpenCV python library. 

## Measuring Keypoint Distances 
Here in this function, we take a list of image paths, and index pairs for images to compare and make a directory to store features.  Here we identify and record correspondences between key points in different images for later alignment.  We do this by looping through pairs of indices to get the images that have been store to compare. 
And then subsequently perform matching to find corresponding keypoints between two images based on their descriptors. 

## Image Reconstruction 

COLMAP is an open source software used for 3D reconstruction from images. It can be  used to build detailed 3D models from individual photographs. In this case we use COLMAPs ingrained function for Exhaustive matching that already makes use of the RANSAC algorithm, that is detailed below :


<img width="452" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/9dc4a95f-3ad4-4cd0-ac17-8dbdaeaa7882">

<img width="385" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/763512bc-325c-4e91-832b-0e8ffb862e75">

Now that we have similar image pairs all mapped into COLMAP, we can use its reconstruction algorithm that starts with two pairs of images and continually adds more images to the scene thus creating a reconstructed scene with camera information.
The final submission must be a CSV file - containing the image_path, the dataset name, the scene, the rotation_matrix and the translation vector. 

## Unique changes I have made to my submission in an attempt to reach a (better/worse) score 

My initial challenge with this competition is familiarizing myself with Kaggle Competitions in general. This contest has a No Internet Clause wherein all models must be pre-downloaded and loaded in a single run. 
This spurred the thought process of using OpenCV library as it has a number of key point detectors and methods to use out of the box without having to download many models. 
This also lead me on to try and see what kind of a baseline score I could reach of my own accord. 

My initial steps would be to replace the detect_keypoints functions by using the suggested traditional methods via OpenCV Python - ORB, SIFT, AKAZE. 

We can try to visualize where the key points are for the following image here 

[INSERT IMAGES HERE]

Next for generating keypoint distances, I go with the traditional, BFMatcher with the cv2.NORM_HAMMING norm type, which is typically good for binary descriptions (like those from AKAZE). This matcher performs brute-force matching with cross-check meaning it ensures mutual matches. Debugging and replacing KF.LightGlueMatcher I noticed that it took a lot more time to calculate the distances observed between key points.

Finally, I could not find a reasonable alternative to PYCOLMAP and the Exhaustive matching algorithm that it uses for reconstruction using the RANSAC algorithm, so I tried to have my keypoints fit the parameter requirements of RANSAC. 
Basically, all the key points are mapped into COLMAP, creating a database. 

The challenge I faced here was keeping track of the shape of the output from SIFT, ORB and AKAZE as opposed to ALIKED descriptors.
The features being generated were of the dimension (,7), when it had to be (,2). This is further reinforced in COLMAP where we can see an **assert(dimension==2)** being implemented in the utility folders. This leads me to assume that the expected features are only x and y. 

However, I would require to go into COLMAPs documentation to further reconfirm that the two features are indeed x,y and not some other inferred feature which is a combination of multiple. 

Due to time and kaggle resource constraints I tried to get my submission scored, yet couldn't due to an exception being thrown at the very end which I'm still trying to figure out. However what I could do is compare the output submission formats and see how well I fared trying to sub out some of the methods used. 





