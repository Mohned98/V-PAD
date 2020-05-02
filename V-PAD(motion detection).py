import cv2
import math
import numpy as np

#define keypad properties
initial_key_x = 30
offset = 60
key_y = 80
font = cv2.FONT_HERSHEY_SIMPLEX
keypad_color = (23,208,253)
keypad_end_line_y = 130

##capture webcam video  
cap = cv2.VideoCapture(0) ## object for the video handle
cap.set(3, 1920) ## change width to 1920 pixels
cap.set(4, 1080) ##change height to 1080 pixels
"""
create a method that can seperate the moving foreground from the background
to detect any moving object through keypad part such as 
- an improved adaptive background mixture model for real-time tracking 
with shadow detection by KaewTraKulPong "cv2.createBackgroundSubtractorMOG"
- an improved adaptive Gaussian mixture model for background subtraction by Zivkovic, 
and Efficient Adaptive Density Estimation per Image Pixel for the Task of Background 
Subtraction, also by Zivkovic "cv2.BackgroundSubtractorMOG2"
"""
##cv2.createBackgroundSubtractorKNN
fgbg = cv2.createBackgroundSubtractorMOG2() ##detectShadows=False for non detecting shadows as grey color
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
##main loop
while True:
    _,frame=cap.read()
    frame = cv2.flip(frame, 1) ## 1 for flipping around the y-axis
    ## draw the keypad on the copy of the original video frame  
    output_image = frame.copy()
    for key in range(10): ##for digits from 0 to 9
        key_x = initial_key_x + key * offset
        cv2.putText(output_image,chr(key+48),(key_x,key_y), font, 1, keypad_color, 3)
    cv2.line(output_image,(0,keypad_end_line_y),(output_image.shape[1],keypad_end_line_y),keypad_color, 2)
    ##make the motion detection happens only on the keypad part
    mask_area = frame[50:keypad_end_line_y, 0:frame.shape[1]]
    detection_mask =  fgbg.apply(mask_area)
    detection_mask = cv2.morphologyEx(detection_mask, cv2.MORPH_OPEN, kernel)
    #find the largest contour in the fgmask image
    _,contours,_ = cv2.findContours(detection_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) ## or cv2.CHAIN_APPROX_NONE
    cv2.drawContours(output_image, contours, -1, (0,255,0), 3)
    ##print(len(contours))
    for cnt in contours:
        (x,y), r = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        print(area)
        if area > 1000 and area < 2000:
            cv2.circle(output_image,(int(x),int(y)),int(r),(23,208,253), thickness=5)
    cv2.imshow('V-PAD',output_image)
    cv2.imshow('Mask',detection_mask)
    key = cv2.waitKey(30) & 0xff
    if key == 27: ## if the key is esc character break the loop then close the video streaming
        break
cap.release()
cv2.destroyAllWindows()

