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

##main loop
while True:
    _,frame=cap.read()
    ## draw the keypad on the copy of the original video frame  
    output_image = frame.copy()
    for key in range(10): ##for digits from 0 to 9
        key_x = initial_key_x + key * offset
        cv2.putText(output_image,chr(key+48),(key_x,key_y), font, 1, keypad_color, 3)
    cv2.line(output_image,(0,keypad_end_line_y),(output_image.shape[1],keypad_end_line_y),keypad_color, 2)
    cv2.imshow('V-PAD',output_image)
    key = cv2.waitKey(30) & 0xff
    if key == 27: ## if the key is esc character break the loop then close the video streaming
        break
cap.release()
cv2.destroyAllWindows()

