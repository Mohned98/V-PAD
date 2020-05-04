import cv2
import numpy as np
import math

adaptive_threshold_max_value = 100
Kernel_size = (3,3)
contour_color = (0,255,0)
circle_point_color = (23,208,253)

# Read a created hand video:
cap = cv2.VideoCapture('hand.avi')
if cap.isOpened() == False:
    print("Error during reading the video")
    exit()

while cap.isOpened():
    # 1.Read each video frame:
    ret,frame = cap.read()
    composite = frame.copy()

    # 2.Apply an image pre-processing:
    # 2.1.Convert each frame to gray scale then make an adaptive threshold:
    gray_image = cv2.cvtColor(composite,cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray_image,adaptive_threshold_max_value,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # 2.2.Make an Open Operation to remove the white noise:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Kernel_size)
    detection_mask = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

    # 3.Find the contours of the largest area (hand Palm):
    contours , hierarchy = cv2.findContours(detection_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(frame,max_contour,-1,contour_color,3)

    # 4.Find the center of the contour:
    moments = cv2.moments(max_contour)
    if moments['m00'] != 0:
      cx = int(moments['m10'] / moments['m00'])
      cy = int(moments['m01'] / moments['m00'])
    center = (cx,cy)
    cv2.circle(frame,center,5,circle_point_color,3)

    cv2.imshow('mask', detection_mask)
    cv2.imshow('hand_detection frame', frame)
    key = cv2.waitKey(1500) & 0xff
    if key == 27:  ## if the key is esc character break the loop then close the video streaming
        break

cap.release()
cv2.destroyAllWindows()