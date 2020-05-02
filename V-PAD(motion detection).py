import cv2
import math
import numpy as np

##capture webcam video  
cap = cv2.VideoCapture(0) ## object for the video handle
cap.set(3, 1920) ## change width to 1920 pixels
cap.set(4, 1080) ##change height to 1080 pixels

##main loop
while True:
    _,frame=cap.read()
    cv2.imshow('V-PAD',frame)
    key = cv2.waitKey(30) & 0xff
    if key == 27: ## if the key is esc character break the loop then close the video streaming
        break
cap.release()
cv2.destroyAllWindows()

