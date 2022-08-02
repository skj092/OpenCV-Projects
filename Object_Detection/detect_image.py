# import the necessary packages 
from torchvision.models import detection
import numpy as np 
import cv2
import torch
import argparse
import pickle
import imutils

# load the list of categories in the COCO dataset and then generate a 
# set of bounding box colors for each class
CLASSES = open(file="coco_classes.txt", mode="rb").read().strip().decode("utf-8")
LABELS = {}
for i, name in enumerate(CLASSES.split("\n")):
    LABELS[i+1] = name.replace("\r", "")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
model.eval()

# load the immage
image = cv2.imread("image.jpg")
image = imutils.resize(image, width=1000)
orig = image.copy()

# convert the image from BGR to RGB channel ordering and change the 
# image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

# add the batch dimensiton, sclle the raw pixel intensities to the range [0, 1],
# and then convert the image to a float tensor
image = np.expand_dims(image, axis=0)
image = image/255.0
image = torch.FloatTensor(image)

# make predictions on the image
detections = model(image)[0]

for i in range(0, len(detections["boxes"])):
    # extrat the confidence (i.e., probability) associated with the prediction
    confidence = detections["scores"][i]
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > 0.5:
        # extract the index of the class label from the detections, 
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections["labels"][i])
        box = detections["boxes"][i].detach().numpy()
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction to our terminal
        label = "{}: {:.2f}%".format(LABELS[idx], confidence * 100)
        print("[INFO] {}".format(label))

        # draw the bounding box and label on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image 
cv2.imwrite('output.png', orig)
cv2.imshow("Output", orig)
cv2.waitKey(0)
