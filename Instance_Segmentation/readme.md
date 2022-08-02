# To make an Image Segmentation model which can extract the human, replace the background and add cap on the head. 

![image](https://user-images.githubusercontent.com/43055935/182149058-0974123a-2f15-4b47-a2c2-f6e1a56e9878.png)



# Plan: 

1. Use Mask RCNN model to find the exact pixal for the human body. 
2. Use OpenCV Haarcascase model to find the coordinates of head.
3. Place image of cap on the head of person. 


# Files:
1. Run msk_rcnn.py to segment human body from the image and replace the background. 

**Input**

<img width="900" alt="face1" src="https://user-images.githubusercontent.com/43055935/182148926-74ba94d1-d8d4-4dea-9bc6-be08db3a30f1.png">

**output**

![extracted](https://user-images.githubusercontent.com/43055935/182149153-c23a580e-5db5-4059-8ffd-e1a057f8836b.png)

2. Run add_cap.py to add Cap on the image with updated background

**Input**

![extracted](https://user-images.githubusercontent.com/43055935/182149153-c23a580e-5db5-4059-8ffd-e1a057f8836b.png)


**output**

![imagewithcap](https://user-images.githubusercontent.com/43055935/182149225-813af151-d1d4-4d9b-85e1-70cb32544d7f.png)


`Its funny how our espection turns into reality :)` 

We have to experiment with different parameters to imporve the result. 
