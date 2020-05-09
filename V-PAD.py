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

def createHandHSVHistogram(frame):
    HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([180, 20, 3], dtype=HSV_frame.dtype) # region of interest (collecting all hand color samples)
    i = 0
    # Looping over the 9 samples of the hand color
    for x in sample_hist_x:
        for y in sample_hist_y:
            # (x0,y0) the first sample beginning coordinates
            x0, y0 = int(x*frame.shape[0]), int(y*frame.shape[1]) + (detection_rec_width // 2) -10
            ROI[i*20 : i*20 + 20, :, :] = HSV_frame[x0 : x0 + 20, y0 : y0 + 20, :]
            i += 1
    """
    form a histogram to represent the frequency of each color appears in the samples 
    using only the Hue and Saturation [0,1] ignoring the third channel (V) which is responsible for 
    the brightness of a color
    """
    handHist = cv2.calcHist([ROI], [0, 1], None, [180, 256], [0, 180, 0, 256]) 
    # normalizi the histogram so we can find the probability of each color being a part of the hand
    return cv2.normalize(handHist, handHist, 0, 255, cv2.NORM_MINMAX)


def histMasking(frame, hand_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # apply the precalculated hand color histogram to capture only the histogram of the skin area of the new image
    back_proj_img = cv2.calcBackProject([hsv_frame], [0, 1], hand_hist, [0, 180, 0, 256], 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim_filter2D, kernel_dim_filter2D))
    cv2.filter2D(back_proj_img, -1, kernel, back_proj_img)
    _, thresh = cv2.threshold(back_proj_img, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_dim_close, kernel_dim_close), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = no_iterations_close)
    
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)


def cropMovingObject(frame):
    detecting_mask =  fgbg.apply(frame, learningRate= bg_sub_learning_rate)
    # eliminate the noise by using morphological operations
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #detecting_mask = cv2.morphologyEx(detecting_mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    detecting_mask = cv2.erode(detecting_mask, kernel, iterations=1)
    detecting_mask = cv2.bitwise_and(frame, frame, mask=detecting_mask)
    return detecting_mask


def binarizeImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(gaussian_blur_dim,gaussian_blur_dim),0)
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh

#capture webcam video  
cap = cv2.VideoCapture(0) # object for the video handle
cap.set(3, 1920) # change width to 1920 pixels
cap.set(4, 1080) #change height to 1080 pixels
cap.set(10, 200) #change brightness to 200
#main loop
while True:
    _,frame=cap.read()
    frame = cv2.flip(frame, 1) # 1 for flipping around the y-axis
    output_image = frame.copy()
    output_image = cv2.bilateralFilter(output_image, 5, 50, 100) #smoothing filter
    # draw the detection rectangle on the copy of the original video frame
    detection_rec_x0 = int(detection_rec_x_start * output_image.shape[1]) # The top left point x value                     
    detection_rec_y0 = int(detection_rec_y_start * output_image.shape[0]) # The top left point y value
    detection_rec_x1 = int(detection_rec_x_end * output_image.shape[1])   # The bottom right point x value
    detection_rec_y1 = int(detection_rec_y_end * output_image.shape[0])   # The bottom right point y value
    detection_rec_height = detection_rec_y1 - detection_rec_y0
    detection_rec_width = detection_rec_x1 - detection_rec_x0
    cv2.rectangle(output_image,(detection_rec_x0,detection_rec_y0),(detection_rec_x1, detection_rec_y1),
                  detection_rec_color, 2)
    if BG_captured == False :
        # then the time hasn't reached yet its limit value so either 
        # the hand histogram hasn't been created yet or 
        # the background subtraction hasn't been performed
        current_time = round(time.perf_counter())
        if (current_time - previous_time) == 1:
            time_in_seconds +=1
            if time_in_seconds == hand_hist_time_limit :
                hand_hist = createHandHSVHistogram(output_image)
                hand_hist_detected = True
            elif time_in_seconds == BG_sub_time_limit:
                time_in_seconds = 0
                """
                create a method that can seperate the moving foreground from the background
                to detect any moving object through keypad part by
                - an improved adaptive Gaussian mixture model for background subtraction by Zivkovic, 
                and Efficient Adaptive Density Estimation per Image Pixel for the Task of Background 
                Subtraction, also by Zivkovic "cv2.BackgroundSubtractorMOG2"
                """
                fgbg = cv2.createBackgroundSubtractorMOG2(0,bg_sub_threshold) 
                BG_captured = True
            print(time_in_seconds)
        previous_time = current_time
    else:
        # then the hand histogram is created and the background subtraction is performed
        roi = output_image[detection_rec_y0:detection_rec_y0 + detection_rec_height,
                           detection_rec_x0:detection_rec_x0 + detection_rec_width]
        # create a mask of the hand using hand color histigram
        roi = cv2.bilateralFilter(roi, 5, 50, 100)
        hist_mask = histMasking(roi, hand_hist)
        hist_mask = binarizeImage(hist_mask)
        cv2.imshow("Histogram Mask",hist_mask)
        # create a mask of the hand using background subtraction
        mov_obj_mask = cropMovingObject(roi)
        mov_obj_mask = binarizeImage(mov_obj_mask)
        cv2.imshow("Motion Mask",mov_obj_mask)
        # and the 2 masks to detect the moving hand
        hand_mask = cv2.bitwise_and(hist_mask, mov_obj_mask)
        cv2.imshow("Hand Mask", hand_mask)
    cv2.imshow('V-PAD',output_image)
    key = cv2.waitKey(30) & 0xff
    if key == 27: # if the key is esc character break the loop then close the video streaming
        break
cap.release()
cv2.destroyAllWindows()

