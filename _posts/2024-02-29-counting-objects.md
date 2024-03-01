
## Measuring Objects in Images and Videos 

## Introduction 

I'm fascinated by videos in which tracking and counting can play a useful role in real life. For instance tracking a puppy through a dog cam to track where it's going.

[![Click to watch the video](http://img.youtube.com/vi/Dj5o17kMBDU/0.jpg)](https://youtu.be/Dj5o17kMBDU)

Or time lapse videos where you can see how objects or people go through different states

[![Click to watch the video](https://img.youtube.com/vi/w77zPAtVTuI/0.jpg)](https://youtu.be/w77zPAtVTuI)


I'm also interested in medical applications,  usually the most direct approaches of using models on images are blood cell count, identifying foreign objects in sample cultures and so on

[![Click to watch the video](https://img.youtube.com/vi/RxHTaTmPlwQ/0.jpg)](https://www.youtube.com/watch?v=RxHTaTmPlwQ)

In this blog I will be focusing on methods I've used to try to isolate and count the number of cells in an image at a time, while also occasionally testing out YOLO on regular home videos to see how the object detection and tracking translates to microscopic videos.

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


   
## Results

## Analysis of Results 


## Analysis of Vision Algorithms

