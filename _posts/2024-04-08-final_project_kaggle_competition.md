

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
I've also tried using ResNet50 to generate embeddings, another custom function to deter from the baseline code. 

```
import torch
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from typing import List

def load_image(path: Path) -> torch.Tensor:
    """Load an image from the disk and convert it to a PyTorch tensor."""
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def embed_images(
    paths: List[Path],
    model_name: str = 'resnet50',
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Computes image embeddings using a pretrained CNN model."""
    model = getattr(models, model_name)(pretrained=True)
    model = model.eval().to(device)

    # Removing the final layer (classification layer)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    embeddings = []

    for path in tqdm(paths, desc="Computing embeddings"):
        image = load_image(path).to(device)
        with torch.no_grad():
            features = model(image)
            # Flatten the features
            embedding = F.normalize(features.view(features.size(0), -1), p=2, dim=1)
        embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = [Path("/content/drive/MyDrive/image_dataset/train/dioscuri/images/3dom_fbk_img_1512.png"), Path("/content/drive/MyDrive/image_dataset/train/dioscuri/images/3dom_fbk_img_1517.png")]
embeddings = embed_images(paths, device=device)

```

## Identifying Image Pairs 
The function get_image_pairs is designed to find pairs of similar images based on the computed embeddings. The approach includes several steps and parameters. 
Here the distance metric is used to calculate distances between all pairs of embeddings using 'torch.cdist'.  Here only pairs with a distance less than the specified 'similarity_threshold' are initially considered as similar. 

We use pre-trained models to get effective embeddings to try and capture the content and style of images, making them suitable for tasks requiring similarity checks for clustering. 

## Detecting keypoints with ALIKED 
ALIKED is a computer vision technique for identifying key points and descriptors in images using a neural network that is designed to be computationally lightweight. Its main innovation is the use of deformable transformations, allowing it to adapt to geometric changes in images, enhancing accuracy in feature extraction when compared to traditional methods like SIFT.

However this is exactly what I will be exploring later on! Using traditional methods like SIFT, ORB and AKAZE that are already available through the OpenCV python library. 


## Detecting keypoints with SuperPoint (Custom written methods with AI help, but lots of tweaking) 

**SUPERPOINT**

Unlike traditional patch-based neural networks, the proposed fully-convolutional model processes full-sized images in one pass, simultaneously identifying pixel-level interest points and generating associated descriptors. The authors develop a technique called Homographic Adaptation, which uses multiple scales and homographies to enhance the repeatability of detected interest points and facilitates adaptation across different domains, such as from synthetic to real images. When trained on the MS-COCO dataset using this method, the model surpasses both an initial pre-adapted deep model and conventional corner detectors like SIFT and ORB, particularly in detecting a broader set of interest points. This results in superior homography estimation performance on the HPatches dataset, achieving state-of-the-art results compared to other methods including LIFT, SIFT, and ORB.

I was able to implement it all the way and even tried to visualize the keypoint matching between multiple images as follows :
The following code was taken from https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py

However I've modified extensively to fit the purpose of the notebook, mainly using the SuperPoint Architecture to load the model directly rather than having to rely on a Python Library

```
import torch
import h5py
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
def load_and_preprocess_image(path, resize_to, device):
    """Loads an image, converts it to grayscale, resizes it, and returns a tensor along with the original size."""
    image = Image.open(path).convert('RGB')
    original_size = image.size  # Save the original size (width, height)
    image = resize(image, (resize_to, resize_to))
    image = to_tensor(image).unsqueeze(0).to(device)
    return image, original_size


def visualize_keypoints(image_path, keypoints, scores=None):
    """Displays an image and overlays the detected keypoints."""
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if scores is not None:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c=scores, cmap='hot', s=10)
        plt.colorbar(label='KeyPoint Scores')
    else:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', s=10)

    plt.axis('off')
    plt.show()


def detect_keypoints(
    paths: list[Path],
    feature_dir: Path,
    num_features: int = 2048,
    resize_to: int = 1024,
    border_margin: int = 50,  # Margin size in pixels after resizing
    device: torch.device = torch.device("cpu"),
    visualize: bool = False
) -> None:
    """Detects keypoints in a list of images using the SuperPoint model, resizes them to original dimensions,
    and stores them, ignoring keypoints close to the borders."""
    dtype = torch.float32
    extractor = SuperPoint(
        max_num_keypoints=num_features,
        detection_threshold=0.01,
        resize=resize_to
    ).eval().to(device)

    feature_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:

        for path in tqdm(paths, desc="Computing keypoints"):
            key = path.stem

            with torch.inference_mode():
                image, original_size = load_and_preprocess_image(path, resize_to, device)
                data = {"image": image}
                features = extractor(data)

                keypoints = features['keypoints'][0].squeeze().detach().cpu().numpy()
                descriptors = features['descriptors'][0].squeeze().detach().cpu().numpy()

                # Filter out keypoints close to the border
                keep = (keypoints[:, 0] >= border_margin) & \
                       (keypoints[:, 0] <= resize_to - border_margin) & \
                       (keypoints[:, 1] >= border_margin) & \
                       (keypoints[:, 1] <= resize_to - border_margin)
                keypoints = keypoints[keep]
                descriptors = descriptors[keep]

                # Rescale keypoints to original dimensions
                scaling_factor_x, scaling_factor_y = original_size[0] / resize_to, original_size[1] / resize_to
                keypoints[:, 0] *= scaling_factor_x
                keypoints[:, 1] *= scaling_factor_y

                f_keypoints.create_dataset(key, data=keypoints)
                f_descriptors.create_dataset(key, data=descriptors)

                if visualize:
                    visualize_keypoints(path, keypoints)

```

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
This spurred the thought process of using the OpenCV library as it has a number of key point detectors and methods to use out of the box without having to download many models. 
This also lead me on to try and see what kind of a baseline score I could reach of my own accord. 

The image embeddings being generated from the baseline notebook are done using an AutoImageProcessor. In an attempt to get different embeddings, I use CLIP, however using it does not generate embeddings that are appropriate for the challenge. Reducing the number of matches significantly. However it is definitely possible to search for models that give better embeddings for that particular image

My initial steps would be to replace the detect_keypoints functions by using the suggested traditional methods via OpenCV Python - ORB, SIFT, AKAZE. 

We can try to visualize where the key points are for the following image here 

**SIFT**

<img width="454" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/89126a50-9d70-4347-b602-964122a0d0f3">

**AKAZE**

<img width="454" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/e592576a-f8be-4c4f-8762-081bdebd2d7f">


**ORB**

<img width="485" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/98053e7f-d045-42f4-94a8-471e85713e18">

**Detections keypoints to beat - ALIKED**

<img width="518" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/00b98a96-6b59-4012-a1b8-abab4e1078f1">



Next for generating keypoint distances, I go with the traditional, BFMatcher with the cv2.NORM_HAMMING norm type, which is typically good for binary descriptions (like those from AKAZE). This matcher performs brute-force matching with cross-check meaning it ensures mutual matches. Debugging and replacing KF.LightGlueMatcher I noticed that it took a lot more time to calculate the distances observed between key points.




**1.Simple Objects**

Transparent Glass object 

<img width="404" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/374be90b-7808-4160-a6bb-f4e1c645e3c7">


Looking at the keypoints SuperPoint detects for each of these simple objects shows that it falls subject to first looking at poor characteristics to outline. For instance initially these were the points that were highlighted:


<img width="390" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/06caa821-26c1-449d-aeaa-47250ed4b9dd">

Here we see borders being highlighted as keypoints, which is not relevant for reconstruction. So to the wrapped of the detect_keypoints I added a border to ignore the keypoints being generated as shown here: 


<img width="386" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/b132acbb-ffd0-4a62-aa97-85d406466434">



To better visualize the matches I iterate over all the images and use:

This is a custom function that I had to implement myself whose header looks like: 

```

from sklearn.neighbors import NearestNeighbors

def mutual_nearest_neighbors(
    paths: list[Path],
    index_pairs: list[tuple[int, int]],
    feature_dir: Path,
    min_matches: int = 15,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
) -> None:
    with h5py.File(feature_dir / "keypoints.h5", "r") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", "r") as f_descriptors, \
         h5py.File(feature_dir / "matches.h5", "w") as f_matches:

        for idx1, idx2 in tqdm(index_pairs, desc="Matching keypoints"):
            key1, key2 = paths[idx1].stem, paths[idx2].stem

            try:
                descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
                descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)
            except KeyError as e:
                print(f"Error accessing descriptors for {key1} or {key2}: {str(e)}")
                continue

            nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(descriptors1.cpu().numpy())
            distances1, indices1 = nbrs1.kneighbors(descriptors2.cpu().numpy())

            nbrs2 = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(descriptors2.cpu().numpy())
            distances2, indices2 = nbrs2.kneighbors(descriptors1.cpu().numpy())

            mutual_matches = []
            for i, j in enumerate(indices1.flatten()):
                if indices2.flatten()[j] == i:
                    mutual_matches.append((i, j))

            mutual_matches = np.array(mutual_matches)

            if n_matches := len(mutual_matches):
                if verbose:
                    print(f"{key1}-{key2}: {n_matches} matches")
                if n_matches >= min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=mutual_matches)

```


<img width="579" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/1f87c8c4-c32d-49f9-9263-bcc6f5e2d790">


**Matching Process:**
Using the sklearn's NearestNeighbors, it computes the nearest neighbor for each descriptor in one image to all descriptors in the other image and vice versa.
Establishes mutual matches by identifying cases where each descriptor is the nearest to the other in both directions (i.e., bidirectional nearest neighbors).


**2.Architectures from far away**

Roman Ruins


<img width="383" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/64e67a95-da7d-4361-a7e7-a0c17e8d15de">


We can try to infer similar results for the Roman Ruins, especially the keypoints and the matching points across different viewing angles:

Keypoints:



<img width="386" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/530cc11e-5ced-4613-8d8b-ab72337bec0a"><img width="384" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/c70a696e-0f37-4a5e-9db6-b2651c79bc48">

Observing the keypoints here, shows that SuperPoint has the tendency to basically outline the shadows of the images, we could try and lower the number of keypoints detected to see if only the key features are detected:


<img width="385" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/b24e9338-2641-4ffa-83dc-5a6767e0e2ed"><img width="382" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/c98c7d4e-9db7-4489-824d-3dc800ab3ac3"><img width="383" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/7e4318c8-d422-4a4c-a1cf-8124ddc32a0e"><img width="286" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/28cef63f-4ccf-4a75-8ed1-c649ea3acc36"><img width="381" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/3d14e040-8570-4391-9eac-27727a61e26f">



The above images are captured with about 1024 keypoints while excluding the faulty border, which shows a generous improvement compared to the earlier iteration

However in some instances there are a few bad matches too, but however this discrepancy is mostly due to the mismatch in image ratios and only visual but by no means is the matching still perfect: 

<img width="574" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/19c5603f-04ca-46d3-8089-93da6987255f"><img width="576" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/49c7a759-5ae2-4ab7-b123-9dd555acaadf">



**3.Complex Structures**

Church


<img width="360" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/490de49a-2dde-43cd-b0be-7ec97d2801a6">


Here we run into problems where the trees take a lot of the keypoints, although we do get good matches like: 

<img width="575" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/e523bdc9-96ee-4ea0-8b38-cb3a4b5fed01"><img width="581" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/2b4ff2b2-a075-489f-a056-8eaf2a280242">

In areas where there is lots of occlusions and trees we get bad matches:


**Trying to Filter out irregularities like hard shapes from the Foreground like Trees:**

However, I try experimenting with different methods to try and ignore trees all together - I attempt this by making a function that essentially crops out any green HSV colors from the original image and saves it another folder. Doing so results in images with matches like so, where you do not see any more matches from the middle of the trees but rather from the outside 



<img width="578" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/3c977211-cde9-42ea-8f34-26ffd75f6dcb">


Bad matches when the green is not masked out and the keypoint extractor recognizes more of the bushes as keypoints leading to improper matching like so: 

<img width="572" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/562ec348-bbf8-4bc5-9380-cad1f0a7e77b">


Finally, I could not find a reasonable alternative to PYCOLMAP and the Exhaustive matching algorithm that it uses for reconstruction using the RANSAC algorithm, so I tried to have my key points fit the parameter requirements of RANSAC. 
Basically, all the key points are mapped into COLMAP, creating a database. 

The challenge I faced here was keeping track of the shape of the output from SIFT, ORB, and AKAZE as opposed to ALIKED descriptors.
The features being generated were of the dimension (,7), when it had to be (,2). This is further reinforced in COLMAP where we can see an **assert(dimension==2)** being implemented in the utility folders. This leads me to assume that the expected features are only x and y.   

However, I would require to go into COLMAPs documentation to further reconfirm that the two features are indeed x, and y and not some other inferred feature which is a combination of multiple. 

Due to time and Kaggle resource constraints, I tried to get my submission scored, yet could not due to an exception being thrown at the very end which I'm still trying to figure out. However, what I could do is compare the output submission formats and see how well I fared trying to sub out some of the methods used. 

Below is the CSV obtained from following the baseline method: 

<img width="545" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/f98a0644-38eb-429d-8d8c-02d0f35646ce">


Of the 41 images required for reconstruction 40 were matched as indicated by this console output 
<img width="352" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/40027a8b-da73-4aad-9a9b-a9c603def68d">

Receiving a score of 0.11

My attempt at this content was able to find only 8 successful reconstructions which definitely shows that matching was not consistent

<img width="506" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/9c89a6b7-514b-4e5d-95b7-47d53d7f4181">


Following are all the matches that are visibly discernable between the images but just counted in lists to see the similarity overview:

<img width="414" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/b889723a-a331-43fe-be14-5ad34829c52f">


I will update this section as soon as I get a reasonable score, but clearly I assume it would be lower, because of the lower matches. 

## Assured Outcomes to Compare to

On the side to see how my reconstruction methods fair against fully built libraries like KF.LightGlueMatcher and papers like Hierarchical Localization, I used the sample datasets in hand to see how the reconstruction fairs here 
https://github.com/cvg/Hierarchical-Localization/


![ezgif-2-a66429bc02](https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/6d89b85f-52ce-443e-a708-2379182fa23e)




<img width="341" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/d73e91d9-9003-4258-9b35-813741a3ceac">



Here we see the keypoint descriptors not ignoring trees, even finding these reconstructions in the 3D model, but surprisingly the distance between different parts of the church are clearly represented here 




## Challenges Faced
**Session Timeouts in Colab**
Often due to memory intensive operations or improper handling of database, after running reconstruction, the entire session in Colab Restarts, or it goes into an endless wait time on Kaggle which could mean that either excess keypoints have been matched and reconstruction is a little difficult to achieve.
**Reconstruction with COLMAP**
Using match_exhaustive, we claim to do Reconstruction with RANSAC algorithm. The sample submission csv format is given by the competition hosts, thus following this format of steps is nececssary. 

## Conclusion 

I believe I can achieve a better performance by looking for deep learning methods used for image embeddings and finding an algorithm that overcomes ALIKED, and LightGlueMatchers. Particularly looking for better generated keypoints in comparison to traditional methods like SIFT, AKAZE, and ORB. Furthermore, there are alternatives for sparse reconstruction as well but I would need to look into algorithms other than RANSAC. The areas of improvement are definitely finding better matches so as to compete with what the baseline model can achieve and then going for reconstruction from there. 










