
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
Source: https://medium.com/@madadihosyn99/generating-adversarial-examples-with-fast-gradient-sign-method-fgsm-in-pytorch-a-step-by-step-a423537628dd
```
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    # Clip the perturbed image values to ensure they stay within the valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image
```





