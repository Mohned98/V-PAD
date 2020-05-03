import cv2
import math
import numpy as np

# 1.define keypad properties:
initial_key_x = 250               ## x coordinate of the first key
offset = 60                      ## used to determine distance between keys
key_y = 80                       ## determines keys y coordinate
font = cv2.FONT_HERSHEY_SIMPLEX
keypad_color = (23,208,253)
hover_color = (18,166,202)       ##keypad keys hover color
keypad_end_line_y = 300
hover_circle_color = (0,120,255) ##circle color that appears when hovering over keypad keys
hover_line_color = (0,0,0)       ##cross color that appear when hovering over keypad keys

# 2.create a string for the entered number:
input_word = ''

# 3.capture webcam video:
cap = cv2.VideoCapture(0)   ## object for the video handle
cap.set(3, 1920)            ## change width to 1920 pixels
cap.set(4, 1080)            ##change height to 1080 pixels
"""
create a method that can seperate the moving foreground from the background
to detect any moving object through keypad part such as 
- an improved adaptive background mixture model for real-time tracking 
with shadow detection by KaewTraKulPong "cv2.createBackgroundSubtractorMOG"
- an improved adaptive Gaussian mixture model for background subtraction by Zivkovic, 
and Efficient Adaptive Density Estimation per Image Pixel for the Task of Background 
Subtraction, also by Zivkovic "cv2.BackgroundSubtractorMOG2"
"""
# 4.cv2.createBackgroundSubtractorKNN:
fgbg = cv2.createBackgroundSubtractorMOG2()            ##detectShadows=False for non detecting shadows as grey color
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 3.main loop:
while True:
    _,frame=cap.read()

    ## 1.for flipping around the y-axis
    frame = cv2.flip(frame, 1)

    ## 2.draw the keypad on the copy of the original video frame
    output_image = frame.copy()
    for key in range(10):   ##for digits from 0 to 9
        key_x = initial_key_x + key * offset
        cv2.putText(output_image,chr(key+48),(key_x,key_y), font, 1, keypad_color, 3)
    cv2.line(output_image,(0,keypad_end_line_y),(output_image.shape[1],keypad_end_line_y),keypad_color, 2)

    ## 3.make the motion detection happens only on the keypad part
    mask_area = frame[50:keypad_end_line_y, 0:frame.shape[1]]
    detection_mask =  fgbg.apply(mask_area)
    detection_mask = cv2.morphologyEx(detection_mask, cv2.MORPH_OPEN, kernel)

                   ## Contour Processing ##
    ######################################################################
    # 4.find the largest contour in the fgmask image:
    contours, hierarchy = cv2.findContours(detection_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  ## or cv2.CHAIN_APPROX_NONE
    if contours:
            max_contour = max(contours,key=cv2.contourArea)
           #cv2.drawContours(output_image, max_contour, -1, (0, 255, 0), 3)

            # 5.Find the center of the contour:
            moments = cv2.moments(max_contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                center = (cx, cy)
            cv2.circle(output_image, center, 2, (0, 255, 0), 3)

            # 5.Find the finger tip using Convex hull:
            hull = cv2.convexHull(max_contour, returnPoints=False)
            convex_defects = cv2.convexityDefects(max_contour, hull)
            if convex_defects is not None:
                for i in range(convex_defects.shape[0]):
                  s, e, f, d = convex_defects[i, 0]
                  start = tuple(max_contour[s][0])
                  end = tuple(max_contour[e][0])
                  far = tuple(max_contour[f][0])
                  cv2.line(output_image, start, end, (0, 255, 0), 2)

    cv2.imshow('V-PAD',output_image)
    cv2.imshow('Mask',detection_mask)
    key = cv2.waitKey(30) & 0xff
    if key == 27: ## if the key is esc character break the loop then close the video streaming
        break

cap.release()
cv2.destroyAllWindows()

