# import the necessary packages
from torchvision.models import detection
import numpy as np
import cv2
import torch
import pickle
import imutils

CLASSES = open(file="coco_classes.txt", mode="rb").read().strip().decode("utf-8")
LABELS = {}
for i, name in enumerate(CLASSES.split("\n")):
    LABELS[i+1] = name.replace("\r", "")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# LOADING THE MODEL
model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
model.eval()

# loading the video 
cap = cv2.VideoCapture(0)

# loop over the frames of the video
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()
    # convert the frame from BGR to RGB channel ordering and
    # then convert the image from channels last to channels first
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    # add the batch dimension, sclle the raw pixel intensities to the range [0, 1],
    # and then convert the image to a float tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame/255.0
    frame = torch.FloatTensor(frame)
    # make predictions on the frame
    detections = model(frame)[0]
    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the prediction
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
            # draw the bounding box and label on the frame
            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # show the output frame
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()