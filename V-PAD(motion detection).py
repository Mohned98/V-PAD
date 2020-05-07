import cv2
import math
import numpy as np
import time

#define keypad properties
initial_key_x = 30 # x coordinate of the first key  
offset = 60 # used to determine distance between keys 
key_y = 80 # determines keys y coordinate
font = cv2.FONT_HERSHEY_SIMPLEX
keypad_color = (23,208,253)
hover_color = (18,166,202) #keypad keys hover color
keypad_end_line_y = 130 
hover_circle_color = (0,120,255) #circle color that appears when hovering over keypad keys
hover_line_color = (0,0,0)  #cross color that appear when hovering over keypad keys

# detection rectangle coordinates percentage of total width and height
detection_rec_x_start = 0.5 
detection_rec_x_end = 1
detection_rec_y_start = 0.0
detection_rec_y_end = 0.8
#detection rectangle color
detection_rec_color = (255, 0, 0)
#variables to calculate elapsed time in seconds
previous_time = 0
time_in_seconds = 0
time_limit = 10
#create a string for the entered number
input_word = ''

BG_captured = False
bgSubThreshold = 50
learning_rate = 0

def cropMovingObject(frame):
    detected_mask =  fgbg.apply(frame, learningRate= learning_rate)
    #eliminate the noise by using morphological operations
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #detected_mask = cv2.morphologyEx(detected_mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    detected_mask = cv2.erode(detected_mask, kernel, iterations=1)
    detected_mask = cv2.bitwise_and(frame, frame, mask=detected_mask)
    return detected_mask

#capture webcam video  
cap = cv2.VideoCapture(0) # object for the video handle
cap.set(3, 1920) # change width to 1920 pixels
cap.set(4, 1080) #change height to 1080 pixels
cap.set(10, 200) #change brightness to 200
#main loop
while True:
    _,frame=cap.read()
    frame = cv2.flip(frame, 1) # 1 for flipping around the y-axis
    # draw the keypad on the copy of the original video frame  
    output_image = frame.copy()
    output_image = cv2.bilateralFilter(output_image, 5, 50, 100) #smoothing filter
    cv2.rectangle(output_image, (int(detection_rec_x_start * output_image.shape[1]),
                                 int(detection_rec_y_start * output_image.shape[0])),
                                (int(detection_rec_x_end * output_image.shape[1]),
                                 int(detection_rec_y_end * output_image.shape[0])),
                                detection_rec_color, 2)
    if BG_captured == False:
        current_time = round(time.perf_counter())
        if (current_time - previous_time) == 1:
            time_in_seconds +=1
            if time_in_seconds >= time_limit :
                time_in_seconds = 0
                """
                create a method that can seperate the moving foreground from the background
                to detect any moving object through keypad part by
                - an improved adaptive Gaussian mixture model for background subtraction by Zivkovic, 
                and Efficient Adaptive Density Estimation per Image Pixel for the Task of Background 
                Subtraction, also by Zivkovic "cv2.BackgroundSubtractorMOG2"
                """
                fgbg = cv2.createBackgroundSubtractorMOG2(0,bgSubThreshold) 
                BG_captured = True
            print(time_in_seconds)
        previous_time = current_time
    else:
        detected_mask = cropMovingObject(output_image)
        detected_mask = detected_mask[int(detection_rec_y_start * output_image.shape[0]):
                                        int(detection_rec_y_end * output_image.shape[0]), 
                                        int(detection_rec_x_start * output_image.shape[1]):
                                        int(detection_rec_x_end * output_image.shape[1])]
        cv2.imshow("Mask", detected_mask)



    cv2.imshow('V-PAD',output_image)
    key = cv2.waitKey(30) & 0xff
    if key == 27: # if the key is esc character break the loop then close the video streaming
        break
cap.release()
cv2.destroyAllWindows()

"""
    for key in range(10): #for digits from 0 to 9
        key_x = initial_key_x + key * offset
        cv2.putText(output_image,chr(key+48),(key_x,key_y), font, 1, keypad_color, 3)
    cv2.line(output_image,(0,keypad_end_line_y),(output_image.shape[1],keypad_end_line_y),keypad_color, 2)
    ###################################
    #find the largest contour in the fgmask image
    _,contours,_ = cv2.findContours(detected_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # or cv2.CHAIN_APPROX_NONE
    cv2.drawContours(output_image, contours, -1, (0,255,0), 3)
    #print(len(contours))
    for cnt in contours:
        (x,y), r = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
"""
