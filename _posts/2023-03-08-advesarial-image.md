
## Adversarial Alchemy: Probing ImageNet's Defenses with Attacks


Here are the various adversarial attacks we will be trying out on the ResNet50 model:

1. Image Compression
2. Blur
3. Tilting
4. Aging the Image
5. Average 2 images together
6. Fast Gradient Sign Method
7. Adversarial Patches

## Image Compression 
Here I've used JPEG compression in the OpenCV library, the code is as follows: 
'''


'''

## Blur
Here I've used Blur on the image, the code is as follows:

The output is as follows:


## Aging the image using a filter
Here I've used the Old age filter from another app to produce a black and white image

## Averaging two images together 
Here I've averaged two unrelated images and averaged them out over a few ratios


## Fast Gradient Sign Method
Here I've experimented with Gradient Sign methods to target another class in the ImageNet labels to try and alter pixels within the image to get the closest to the false class 




