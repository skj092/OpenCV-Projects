# importing the necessary packages
import os
import numpy as np
import random 
import time 
import cv2 
import argparse
from imutils.video import VideoStream
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask_rcnn", required=True, help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
ap.add_argument("-k", "--kernel", type=int, default=41, help="kernel size for blur")
args = vars(ap.parse_args())

# load the COCO class labels and Mask R-CNN model 
labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# load our Mask R-CNN trained model and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"], "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our mask R-CNN trained model on the COCO dataset
print("[INFO] loading Mask R-CNN model...")
model = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# construct the kernel for the Gaussian blur and initialize whether or not we are in "privacy mode"
k = (args["kernel"], args["kernel"])
privacy = False

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threshold video stream
    frame = vs.read()

    # resize the frame to have a maximum width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    frame = imutils.resize(frame, width=600)
    (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    model.setInput(blob)
    (boxes, masks) = model.forward(["detection_out_final", "detection_masks"])
    print( "passes through the network")
    print("model", model)

    # sort the indexes of the bounding boxes in by their corresponding 
    # prediction probability (in descending order)
    idxs = np.argsort(boxes[0, 0, :, 2])[::-1]

    # initialize the mask, ROI, and coordinates of the person for the 
    # current frame
    mask = None
    roi = None
    coords = None

    # loop over th eindexes
    print("loop over the indexes")
    for i in idxs:
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # if the detection is not the "person" class, ignore it
        if LABELS[classID] != "person":
            continue

        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            coords = (startX, startY, endX, endY)
            boxW = endX - startX
            boxH = endY - startY
            # extract the pixel wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_AREA)
            mask = (mask > args["threshold"])

            # extract the ROI and break from the loop (since we make 
            # the assumption there is only *one* person in the frame 
            # who is also the person with the highest prediction 
            # confidence)
            roi = frame[startY:endY, startX:endX]
            break
    # innitialize our output frame
    print('initialize our output frame')
    output = frame.copy()

    # if the mask is not None, *and* we are in privacy mode, then 
    # we know we can apply the mask and ROI to the output image
    if mask is not None and privacy:
        # blur the ourput frame
        output = cv2.GaussianBlur(output, k, 0)

        # akk the ROI to the output frame for only the masked region
        (startX, startY, endX, endY) = coords
        output[startY:endY, startX:endX][mask] = roi

    # show the output frame
    print('show the output frame')
    cv2.imshow("final output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the 'p' key is pressed, then we are going to "privacy mode"
    if key == ord("p"):
        privacy = not privacy
        print("[INFO] privacy mode: {}".format(privacy))
    # if the q key was pressed, break from the loop
    elif key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()