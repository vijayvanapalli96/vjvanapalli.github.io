
## Measuring Growth Rate in Objects

## Introduction 

In this blog post we will be exploring different methods to explore the growth of objects, or the changes that appear in them over time. 

This post will mainly try to explore these changes through two methods:

1. Segment Anything
2. Preprocessing each frame to highlight the objects on the video, and performing contour detection

## Segment Anything 



   
## Preprocessing Images

This process proved to be less computationally intensive.
For the same video, I thresholded the image, and used cv2s findContour function to isolate the foreground object from the blank background. The contrasting colors and the high definition video allowed me to not account of any occlusion or noise. 
The results can be seen below



In both ac
## Analysis of Results 

From the results above we can see it usually is just a toss between the DoG filters, as they are more consistent atleast with finding blobs. Another unexplored alternative would be to try out a tensorflow model, trained on detecting the actual blobs to see how this fairs against the Vision Algorithms deployed here. 

## Analysis of Vision Algorithms
These Vision algorithms are quite fun to tweak with, as they allow you to narrow down the type of blob that you are searching for. If I wanted to target the larger white blood cells I would simply have to increase the size of the blob and zero in on the defects, however it is a nuanced parameter as red blood cells can also have large sizes, so it is not appropriate to just distinguish based on size. 
Training a model to recognize the color, shape and collection of the blood cells would be a more long standing solution. 
