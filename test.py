# import the necessary modules
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# define the lower and upper boundaries of the "yellow"
# object in the HSV color space, then initialize the
# list of tracked points
yellowLower = (20, 100, 100)
yellowUpper = (30, 255, 255)
pts = deque(maxlen=64)

# load the image 
image = cv2.imread("img.jpg")
# resize the image to a smaller factor so that
# the shapes can be approximated better
image = imutils.resize(image, width=600)
# selecting the color of the object to be tracked
yellowLower = (20, 100, 100)
yellowUpper = (30, 255, 255)
# convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# construct a mask for the color "yellow", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, yellowLower, yellowUpper)
# prining wheter mask is not zero 
print(mask[np.where(mask != 0)])
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# find contours in the mask and initialize the current
# (x, y) center of the cup
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
center = None
# only proceed if at least one contour was found
if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # only proceed if the radius meets a minimum size
    if radius > 10:
        # draw the circle and centroid on the frame,
        cv2.circle(image, (int(x), int(y)), int(radius),
                     (0, 255, 255), 2)
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        # view the image and wait for a keypress
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        
