# Object Tracking 

In this project we will build a simple object tracking system.
We will use a webcam to capture images of a object and then we will
use OpenCV to find the object in the image and track it.



## Result: 

![](tracked.gif)


## Steps:
1. Load the video 
2. Convert each frame to HSV
3. Use mask to select object of interest
4. Grab the coordinates of the center of the object
5. Draw a circle around the object
6. Draw a line from the center of the object in frame i to the center of the object in frame i+1 

