import cv2
import numpy as np




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

# Detection Rectangle properties
# detection rectangle coordinates percentage of total width and height
detection_rec_x_start = 0.5 
detection_rec_x_end = 1
detection_rec_y_start = 0.1
detection_rec_y_end = 0.8
# detection rectangle color
detection_rec_color = (255, 0, 0)

# tuning parameters
# Mothion detection parameters
bg_sub_threshold = 50 # threshold value of the background subtractor function
bg_sub_learning_rate = 0
gaussian_blur_dim = 41 # GaussianBlur kernel parameter
threshold = 60 # Binary threshold
# Hand color detection parameters
kernel_dim_filter2D = 21 # kernel dimension of 2D filter for masking hand color
kernel_dim_close = 5 # kernel dimension of morphological close operation 
no_iterations_close = 7 # number of iteration of morphological close operation

# variables to calculate elapsed time in seconds
previous_time = 0
time_in_seconds = 0
hand_hist_time_limit = 10 # time to perform capture samples of hand color and perform calculate its histogram
BG_sub_time_limit = 20    # time to perform background subtraction action

# hand histogram samples coordinates
sample_hist_x = [6.0/20.0, 9.0/20.0, 12.0/20.0]
sample_hist_y = [9.0/20.0, 10.0/20.0, 11.0/20.0]

BG_captured = False
hand_hist_detected = False


# create a string for the entered number
input_word = ''

#capture webcam video  
cap = cv2.VideoCapture(0) # object for the video handle
cap.set(3, 1920) # change width to 1920 pixels
cap.set(4, 1080) #change height to 1080 pixels
cap.set(10, 200) #change brightness to 200