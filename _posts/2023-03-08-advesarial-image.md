
## Adversarial Alchemy: Probing ImageNet's Defenses with Attacks


Here are the various adversarial attacks we will be trying out on the ResNet50 model:

1. Image Compression
2. Blur
3. Average 2 images together
4. Fast Gradient Sign Method

## Image Compression 
Here I've used JPEG compression in the OpenCV library, the code is as follows: 
```
from PIL import Image

def compress_image(input_path, output_path, quality=85):
    """
    Compress a JPEG image.

    Parameters:
    - input_path (str): Path to the input image.
    - output_path (str): Path to save the compressed image.
    - quality (int): Quality level for compression (0 to 100).

    Returns:
    - None
    """
    try:
        # Open the image
        img = Image.open(input_path)

        # Save the compressed image with the specified quality
        img.save(output_path, 'JPEG', quality=quality, optimize=True)

        print(f"Compression completed successfully. Image saved at {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file_path = "puppy.jpg"
output_file_path = "output_compressed_image.jpg"
compress_image(input_file_path, output_file_path, quality=100)
```
I don't observe any immediate changes in class, it remains the same so I explore more extreme methods of achieving image compression, disregarding the quality of the image. 

Following is the output label obtained on an extremely compressed image. Comparing the sizes of the image, the original was 23 KB and the final is 3 KB

<img width="224" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/39686ce1-f4d2-49d0-a3d4-68b40dd17622">


Here the label corresponds to Kuvasz which is another pale white dog breed



## Blur
With a strong blur applied through photo editing, we see the following label occur 

<img width="245" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/4c2056a6-fa6b-4476-8cf2-41cf42c406d8">

The output label is a Hussar monkey which is far removed from the object of the current photo which is a puppy.
To our human perception, despite the blur, we can still tell that the image being displayed is that of a puppy.


## Averaging two images together 
Here I've averaged two unrelated images and averaged them out over a few ratios

The following function was used to blend the two images together: 
```
from PIL import Image
import numpy as np

def blend_images(image_path1, image_path2, output_path, ratio=0.5):
    """
    Blend two images based on a specified ratio and save the result.

    Parameters:
    - image_path1 (str): Path to the first image.
    - image_path2 (str): Path to the second image.
    - output_path (str): Path to save the blended image.
    - ratio (float): Mixing ratio for the combination (0.0 to 1.0).

    Returns:
    - None
    """
    try:
        # Open the images
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        # Resize the second image to match the dimensions of the first image
        img2 = img2.resize(img1.size)

        # Convert images to NumPy arrays for easier manipulation
        array1 = np.array(img1.convert("RGB"))
        array2 = np.array(img2.convert("RGB"))

        # Blend the pixel values based on the specified ratio
        result_array = (array1 * ratio + array2 * (1 - ratio)).astype("uint8")

        # Convert the result array back to an image
        result_img = Image.fromarray(result_array)

        # Save the blended image
        result_img.save(output_path)

        print(f"Blending completed successfully. Image saved at {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage with a 70:30 ratio (you can adjust the ratio as needed)
image_path1 = "puppy.jpg"
image_path2 = "kia.jpg"
output_path = "blended_image.jpg"
blend_images(image_path1, image_path2, output_path, ratio=0.7)

```
Keeping a ratio of 0.7, we observe the following output


<img width="236" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/08086b2c-d6db-4632-a47a-6c74760c9229">


Looking up the label on Imagenet we see the result of 'diaper,nappy,napkin'. 

Experimenting with different ratios yields different results.

Below is the image ratio of 0.5

<img width="217" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/253e591b-d98f-40da-8714-01b6615b112a">





Keeping an equal ratio, provides the label 'Oxygen mask', which shows how erratically the feature map being generated from these images affects the final label


To average the image with a more anatomically similar object, I use a bear facing the camera in a similar orientation to see when the label shifts.
Around a ratio of 0.6 is when the classifier starts to distinguish between a puppy and a bear. 

The following is the desired brown bear label classification

<img width="257" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/ef7f6677-5d14-41a9-b66b-b6ba619a17e7">






## Fast Gradient Sign MethodThe 
Here I've experimented with Gradient Sign methods to target another class in the ImageNet labels to try and alter pixels within the image to get the closest to the false class 
Source: https://medium.com/@madadihosyn99/generating-adversarial-examples-with-fast-gradient-sign-method-fgsm-in-pytorch-a-step-by-step-a423537628dd

The function is as follows: 
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
The gist of what I understand here is that it is implementing a gradient that tweaks pixel values toward a different class.

We can observe the following output:

<img width="222" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/ecb3317f-4ef8-4684-b4ef-e33830e6d877">

Looking at the labels in Imagenet, we see that the original is a golden retriever (207)
Whereas
the new image is a Komondor which is a  drastically different dog breed (228)







