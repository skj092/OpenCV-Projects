# Object Tracking 

In this project we will build a simple object tracking system.
We will use a webcam to capture images of a ball and then we will
use OpenCV to find the ball in the image and track it.



## Result: 

![](tracked.gif)


## Steps:
1. Load the video 
2. Convert each frame to HSV
3. Grab the coordinates of the center of the ball
4. Draw a circle around the ball
5. Draw a line from the center of the ball in frame i to the center of the ball in frame i+1 

