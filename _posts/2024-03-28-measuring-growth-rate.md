
## Measuring Growth Rate in Objects

## Introduction 

In this blog post we will be exploring different methods to explore the growth of objects, or the changes that appear in them over time. 

This post will mainly try to explore these changes through two methods:

1. Segment Anything
2. Preprocessing each frame to highlight the objects on the video, and performing contour detection

## Method

Examining how Yolo works on regular videos, we see that it's quite good at detecting dogs, cats, furniture, except for when the dog comes really close to the camera. The change in perspective and the focus on partial face confuses the YOLOv8 model. 

[![Click to watch the video](https://img.youtube.com/vi/fHGpIKpM9c4/0.jpg)](https://www.youtube.com/watch?v=fHGpIKpM9c4)

Similarly using the YOLOv8 model on tracking the growth of a plant was an interesting idea. However, the model misclassifies the plant as a bird but keeps a reasonably tight bounding box on the object of interest. 
Initially I aimed to take a measure of the width of the bounding box and see how that grows  along with the duration of the video. Similarily, the height could also be plotted down. 


[![Click to watch the video](https://img.youtube.com/vi/iS-svnWWQmo/0.jpg)](https://www.youtube.com/watch?v=iS-svnWWQmo)

However the misclassification is way too inconsistent, also there is a brief period where the model refuses to find the plant in question, so we cant exactly take measurements.
Regardless below are plots of the varying width and height during the short durations of the growth stages displayed. 



On the other hand for a close up of red blood cell video, I first attempted to count the number of blobs on screen, by using OpenCVs simple blob detector. The result is as follows: 


[![Click to watch the video](https://img.youtube.com/vi/RrlAazsjsIE/0.jpg)](https://www.youtube.com/shorts/RrlAazsjsIE)

Here in the top left corner, you can see a count of the blobs that I've labeled conveniently as red blood cells, but in reality, any blob that would satisfy the area requirement would have added to the count. 

This led to an interesting dive of blob detection algorithms. 
A quick overview or the reason the following filters are used is explained as follows, along with demos of the videos using those filters. 

1. LoG Filter (Laplacian of Gaussian)
  Purpose: Detects edges and fine details in the image by applying the Laplacian Operater followed by Gaussian smoothing

[![Click to watch on YouTube](https://img.youtube.com/vi/ux8EbZzRGUw/0.jpg)](https://youtube.com/shorts/ux8EbZzRGUw)


3. DoG Filter (Difference of Gaussian)
   Purpose: Enhance the perception of edges and details at different scales by subtracting one blurred version of the image from another to create a bandpass filter

   [![Click to watch on YouTube](https://img.youtube.com/vi/GdNdzePEB18/0.jpg)](https://youtube.com/shorts/GdNdzePEB18)

    

   
5. DoH Filter (Determinant of the Hessian)
    Purpose: Detect blobs or regions with significant intensity variations by computing the Hessian matrix and calculates the Determinant

   [![Click to watch on YouTube](https://img.youtube.com/vi/x1C6DObc73w/0.jpg)](https://youtube.com/shorts/x1C6DObc73w?feature=share)

7. Blob Detection (SimpleBlobDetector)
   Purpose: Identifies and locates regions of interest using the SimpleBlobDetector in OpenCV

   [![Click to watch on YouTube](https://img.youtube.com/vi/4wuzKxqZRDs/0.jpg)](https://youtube.com/shorts/4wuzKxqZRDs?feature=share)

   The simpleBlobDetector is more consistent in its output especially when the camera is more still



   
## Results

Plotting the data obtained from the filters used above we obtain: 

<img width="518" alt="image" src="https://github.com/vijayvanapalli96/vjvanapalli.github.io/assets/46009628/40d30026-99eb-4107-aad0-4480f1d40293">


## Analysis of Results 

From the results above we can see it usually is just a toss between the DoG filters, as they are more consistent atleast with finding blobs. Another unexplored alternative would be to try out a tensorflow model, trained on detecting the actual blobs to see how this fairs against the Vision Algorithms deployed here. 

## Analysis of Vision Algorithms
These Vision algorithms are quite fun to tweak with, as they allow you to narrow down the type of blob that you are searching for. If I wanted to target the larger white blood cells I would simply have to increase the size of the blob and zero in on the defects, however it is a nuanced parameter as red blood cells can also have large sizes, so it is not appropriate to just distinguish based on size. 
Training a model to recognize the color, shape and collection of the blood cells would be a more long standing solution. 
