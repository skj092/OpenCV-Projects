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
roi = image[cap_H:cap_H+h, cap_W:cap_W+w]
cv2.imshow("ROI", roi)
cv2.imshow("Image", image)
cv2.waitKey(0)