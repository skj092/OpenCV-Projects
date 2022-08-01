import os 
import cv2
import numpy as np
import random
import time
import imutils 
from imutils.video import VideoStream


# load the COCO class labels
labelPath = os.path.join("mask-rcnn-coco", "object_detection_classes_coco.txt")
LABELS = open(labelPath).read().strip().split("\n")

# laod our Mask R-CNN trained model and model configuration
weightsPath = os.path.join("mask-rcnn-coco", "frozen_inference_graph.pb")
configPath = os.path.join("mask-rcnn-coco", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
# load our mask R-CNN trained model on the COCO dataset
print("[INFO] loading Mask R-CNN model...")
model = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load the image 
image = cv2.imread("images/face1.webp")

# resize the image to have a maximum width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = imutils.resize(image, width=600)
(H, W) = image.shape[:2]

# construct a blob from the input frame and then perform a forward
# pass of the Mask R-CNN, giving us (1) the bounding box coordinates
# of the objects in the image along with (2) the pixel-wise segmentation
# for each specific object
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
model.setInput(blob)
(boxes, masks) = model.forward(["detection_out_final", "detection_masks"])

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
    # extract the class ID of the detection along with the confidence
    # (i.e., probability) associated with the prediction
    classID = int(boxes[0, 0, i, 1])
    confidence = boxes[0, 0, i, 2]

    # if the detection is not the "person" class, ignore it
    if LABELS[classID] != "person":
        continue

    # filter out weak predictions by ensuring the detected probability
    # is greater than the minimum probability
    if confidence > 0.5:
        # show the class label 
        print("[INFO] showing output for class {}".format(LABELS[classID]))

        # scale the bounding box coordinates back relative to the
        # size of the image, and then compute the width and the height
        # of the bounding box
        (H, W) = image.shape[:2]
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        boxW = endX - startX
        boxH = endY - startY
        # extract the pixel-wise segmentation for the object, resize
        # the mask such that it's the same dimensions of the bounding
        # box, and then finally threshold to create a *binary* mask
        mask = masks[i, classID]
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_AREA)
        mask = (mask > 0.5).astype("uint8") * 255

        # allocate memory for the box mask and then perform a
        # bitwise AND on the mask and the image to extract the
        # specific object
        boxMask = np.zeros(image.shape[:2], dtype="uint8")
        boxMask[startY:endY, startX:endX] = mask
        extracted = cv2.bitwise_and(image, image, mask=boxMask)

        # adding virtal background to the extracted object

        # loading background image 
        bg = cv2.imread("images/beach.jpg")
        # resizing background image to the same size as extracted object
        w, h = extracted.shape[:2]
        bg = cv2.resize(bg, (h, w), interpolation=cv2.INTER_AREA)

        # adding background image to the extracted object
        inverse = cv2.bitwise_not(boxMask)
        bg = cv2.bitwise_and(bg, bg, mask=inverse)
        final = cv2.add(extracted, bg)

        # drawing bounding box on the extracted object
        # cv2.rectangle(final, (startX, startY), (endX, endY), (0, 255, 0), 2)
        print("location of object: ", startX, startY, endX, endY)

        cv2.imwrite("output/extracted.png", final)
        cv2.imshow("Image", image)
        cv2.imshow("FINAL result", final)
        cv2.waitKey(0)