# in this python script we add a cap to a given image using opencv
# and save it to a new file

# import the necessary packages
import cv2
import numpy as np

# load the image
image = cv2.imread("images/final.png")

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# loading face cascase from opencv
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(blurred, 1.3, 5)

# print the number of faces detected
print(len(faces))


