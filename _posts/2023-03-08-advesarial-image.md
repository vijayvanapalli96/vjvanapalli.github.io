
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
Here I've used Blur on the image, the code is as follows:

The output is as follows:


## Aging the image using a filter
Here I've used the Old age filter from another app to produce a black and white image

## Averaging two images together 
Here I've averaged two unrelated images and averaged them out over a few ratios


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







