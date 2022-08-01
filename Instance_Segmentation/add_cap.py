# adding image of cap to the final image
# 122 0 372 184

import cv2
import numpy as np
import os

# load the image
image = cv2.imread("output/extracted.png")
cap = cv2.imread("images/cap.jfif")
cap_W, cap_H = cap.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loading the haarcascade classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
faces_coord = face_cascade.detectMultiScale(gray, 1.3, 5)

def draw(img,faces_coord):
    for (x, y, w, h) in faces_coord:
        # To make a face blurred
        ROI = img[y:y+h, x:x+w]
        # To make a bounding box #*(Not Necessary)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)


x, y, w, h = faces_coord[0]
img = image.copy() 
draw(img,faces_coord)
face = image[y:y+h, x:x+w]
# selecting the position for the cap
cap_region = image[0:50, 172-50:172+130]

# cap image
h,w = cap_region.shape[:2]
resized_cap = cv2.resize(cap, (w,h), interpolation = cv2.INTER_AREA)

# create a mask of cap and inverted mask of the cap
img2gray = cv2.cvtColor(resized_cap, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# black out the area of cap in the image
img1_bg = cv2.bitwise_and(cap_region, cap_region, mask = mask)
# Take only region of image1 from region of image2
img2_fg = cv2.bitwise_and(resized_cap, resized_cap, mask = mask_inv)
# Put image1 on image2
dst = cv2.add(img1_bg, img2_fg)
image[0:50, 172-50:172+130] = dst
cv2.imwrite("output/imagewithcap.png", image)
cv2.imshow("Image with Cap", image)
cv2.waitKey(0)