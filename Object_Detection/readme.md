# AIM:  To Build an Object Detection Model which can able to detect Object in an Image or Video

## Requirements:
1. coco_classes.txt - contain all the classes of coco dataset 
2. torch, and torchvision libraries 

## Model: 
We have used faster RCNN pretrained model to detect object in the image or the video


## Proceture:
1. Install and Import the necessary libraries
2. Define Id2Label dict for COCO label
3. Load the pretrained Faster R-CNN pytorch model
4. Load the image or video frame and preprocess so that it is appropriate for model
5. Predict the output
6. For confidence > 0.5 draw the bouding box and label on the image or video frame.

## Observations:
1. While predicting on video frame it is lagging, 

## Conclusion:
1. We can use YOLO for fast detection on live videostream. 

## Result:
![image](https://user-images.githubusercontent.com/43055935/182361084-c8565ffd-e22f-43f2-8339-da3271dc6f13.jpg)
![output](https://user-images.githubusercontent.com/43055935/182362032-8f90c008-b13b-4176-9889-dd1e351ae673.png)