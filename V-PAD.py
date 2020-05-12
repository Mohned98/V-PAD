import cv2
import math
import numpy as np
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.font as tkFont

# define keypad properties:
rec_key_width = 100                   # the width of the key rectangle
rec_key_height = 80                   # the height of the key rectangle
rec_key_y = 70                        # start y position of key rectangle
key_rectangle_positions = []          # list to store the positions of each key rectangle
key_value_positions = []              # list to store the positions of the key itself
pressed_key_buffer = []               # buffer to store the pressed key
key_actions_dx = 40                   # horizontal distance of the action rectangle from the original keypad
key_actions_dy = 350                  # vertical distance between the action keys rectangles
keypad_distance_from_border = 70      # horizontal distance between rectangle border and keypad
font = cv2.FONT_HERSHEY_SIMPLEX       # Font type
keypad_color = (23,208,253)           # Keypad border and keys color
hover_color = (255,0,0)               # keypad keys hover color
hover_circle_color = (0,200,0)        # circle color that appears when hovering over keypad keys
hover_line_color = (0,0,0)            # cross color that appear when hovering over keypad keys
hover_rectangle_color = (255,255,255) # rectangle key hover color 
Enter_button_color = (0,255,0)        # actions buttons color
Cancel_button_color = (0,0,255)
Clear_button_color = (0,255,255)

# Hand Histogram detection rectangle properties
hand_hist_rec_color = (0, 255, 0)
# hand histogram samples coordinates
sample_hist_x = [6.0/20.0, 9.0/20.0, 12.0/20.0]
sample_hist_y = [9.0/20.0, 10.0/20.0, 11.0/20.0]

# Detection Rectangle properties
# detection rectangle coordinates percentage of total width and height
detection_rec_x_start = 0.5 
detection_rec_x_end = 0.98
detection_rec_y_start = 0.1
detection_rec_y_end = 0.8
# detection rectangle color
detection_rec_color = (255, 0, 0)

# Withdraw and Deposit buttons coordinates and properties:
button_width = 170
button_height = 80
xdistance_between_buttons = 120
height_from_txt = 170
button_color = (54,38,255)

# Variables to calculate the finger tip point:
previous_fingertip_point = (0,0)
farthest_x_margin = 80
closest_x_margin = 4

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
hand_hist_time_limit = 4 # time to perform capture samples of hand color and perform calculate its histogram
BG_sub_time_limit = 8   # time to perform background subtraction action

# Number of milliseconds the welcome page waits
welPage_delay = 2000

fgbg = None      # for foreground subtraction handling
hand_hist = None # for calculating hand histigram

# Phases indication flags 
BG_captured = False
hand_hist_detected = False

# action of the pressed button 'Withdraw' , 'Deposit' , 'Cancel' , 'Clear' or 'Enter'
button_action = ''
# create a string for the entered number
input_word = ''

#Saved Password
password='7854'

#flag for submitting password
password_entered=FALSE

#current page for user (PasswordPage=1, ChooseService=2, Deposit=3, Withdraw=4, Inquiry=5)
currentPage = 1

#Flag for completing deposit
deposit_done=FALSE

#Flag for submitting the amount of money to be withdrawed
money_entered=FALSE

#Available money balance for client
Balance=500000
    

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


def histMasking(frame, hand_histo):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # apply the precalculated hand color histogram to capture only the histogram of the skin area of the new image
    back_proj_img = cv2.calcBackProject([hsv_frame], [0, 1], hand_histo, [0, 180, 0, 256], 1)

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

def draw_keypad_background(frame):
    key_rec_x0 = int(detection_rec_x_start * frame.shape[1]) + keypad_distance_from_border
    key_rec_y0 = rec_key_y
    key_rec_x1 = key_rec_x0 + (rec_key_width * 3)
    key_rec_y1 = key_rec_y0 + (rec_key_height * 4)
    cv2.rectangle(frame,(key_rec_x0,key_rec_y0),(key_rec_x1,key_rec_y1),hover_line_color,-1)

def draw_keypad(frame):
    key_num = 0
    key_actions = ['Cancel','Clear','Enter']
    last_rows_keys = ['','0','']
    if (len(key_rectangle_positions) != 0) and (len(key_value_positions) != 0):
        key_value_positions.clear()
        key_rectangle_positions.clear()

    draw_keypad_background(frame)
    # draw the keypad number keys:
    for row in range (4):
        key_rec_x0 = int(detection_rec_x_start * frame.shape[1]) + keypad_distance_from_border
        key_rec_y0 = rec_key_y + row * rec_key_height
        key_rec_x1 = key_rec_x0 + rec_key_width
        key_rec_y1 = key_rec_y0 + rec_key_height
        for col in range (3):
           # the key itself coordinates inside the rectangle:
           key_x_p = key_rec_x0 + int(rec_key_width / 3)
           key_y_p = key_rec_y0 + int(rec_key_height / 1.5)

           # draw the rectangle of the key and the key:
           if (row == 3):
              key_num = last_rows_keys[col]
           else:
              key_num += 1
           cv2.rectangle(frame, (key_rec_x0, key_rec_y0), (key_rec_x1, key_rec_y1), keypad_color, 5)
           cv2.putText(frame, str(key_num), (key_x_p, key_y_p), font, 1.5, keypad_color, 3)

           # store the key num and its rectangle coordinates:
           key_value_positions.append([key_x_p, key_y_p, str(key_num)])
           key_rectangle_positions.append([(key_rec_x0, key_rec_y0), (key_rec_x1, key_rec_y1), str(key_num)])

           # Update the x and y position of the rectangle:
           key_rec_x0 = key_rec_x1
           key_rec_x1 = key_rec_x0 + rec_key_width
    
    key_rec_x0 = int(detection_rec_x_start * frame.shape[1]) + keypad_distance_from_border - 30
    # draw the key actions:
    for i in range (3):
        # the rectangle coordinates of the key actions:
        action_rec_x0 = key_rec_x0 + i * (key_actions_dx + rec_key_width)
        action_rec_y0 = rec_key_y + key_actions_dy
        action_rec_x1 = action_rec_x0 + rec_key_width - 10
        action_rec_y1 = action_rec_y0 + rec_key_height - 20

        # The action key coordinates:
        action_x_p = action_rec_x0 + int(rec_key_width / 4)
        action_y_p = action_rec_y0 + int(rec_key_height / 2)

        # Action key color:
        if (i == 0):
            color = Cancel_button_color
        elif(i == 1):
            color = Clear_button_color
        else:
            color = Enter_button_color

        # draw the actions and its rectangle:
        cv2.rectangle(frame, (action_rec_x0, action_rec_y0), (action_rec_x1, action_rec_y1), color, -1)
        cv2.putText(frame, key_actions[i], (action_x_p - 17 , action_y_p), font, 0.75, (0,0,0), 2)

        # store the actions and its rectangle coordinates:
        key_value_positions.append([action_x_p - 17, action_y_p, key_actions[i]])
        key_rectangle_positions.append([(action_rec_x0,action_rec_y0), (action_rec_x1,action_rec_y1), key_actions[i]])

def draw_deposit_withdraw_buttons(frame):
    if (len(key_rectangle_positions) != 0) and (len(key_value_positions) != 0):
        key_value_positions.clear()
        key_rectangle_positions.clear()

    txt_point = (int(detection_rec_x_start * frame.shape[1]) + 45, rec_key_y + 100)
    cv2.putText(frame,"Payment Process",txt_point,font,1.5,hover_line_color,3)

    rec_x0 = int(detection_rec_x_start * frame.shape[1]) + 20
    rec_y0 = rec_key_y + height_from_txt
    action_keys = ['Withdraw','Deposit']
    for i in range (2):
       # The action key coordinates
       action_x_p = rec_x0 + int(rec_key_width / 3)
       action_y_p = rec_y0 + int(rec_key_height / 1.5)

       # draw the buttons
       cv2.rectangle(frame,(rec_x0,rec_y0),(rec_x0 + button_width,rec_y0 + button_height),button_color,-1)
       cv2.putText(frame, action_keys[i], (action_x_p - 10, action_y_p), font, 1, hover_line_color, 2)

       # Store the key and its rectangle coordinates
       key_rectangle_positions.append([(rec_x0,rec_y0),(rec_x0 + button_width,rec_y0 + button_height),action_keys[i]])
       key_value_positions.append([action_x_p,action_y_p,action_keys[i]])
       rec_x0 = rec_x0 + button_width + (xdistance_between_buttons)

def contour_centroid(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx,cy
    else:
        return None

def get_fingertip (defects_start_points,contour,centroid,frame_shape):
    farthest_points = []
    cx = centroid[0] ; cy = centroid[1]     # center point coordinates of hand contour:
    finger_tip_x = 0;  finger_tip_y = 0

    # retrieve the x and y coordinates of the defects start points:
    x = np.array(contour[defects_start_points][:, 0][:, 0], dtype=np.float)
    y = np.array(contour[defects_start_points][:, 0][:, 1], dtype=np.float)

    # calculate the euclidean distance from the centre to start points:
    Xpoints_subtract_Xcenter = cv2.pow(cv2.subtract(x, cx), 2)
    Ypoints_subtract_Ycenter = cv2.pow(cv2.subtract(y, cy), 2)
    distance = cv2.sqrt(cv2.add(Xpoints_subtract_Xcenter, Ypoints_subtract_Ycenter))
    max_distance_index = np.argmax(distance)              # fingertip point locates at the most distance from the center
    if max_distance_index < len(defects_start_points):
        finger_tip_index = defects_start_points[max_distance_index]
        finger_tip_x = contour[finger_tip_index][0][0]
        finger_tip_y = contour[finger_tip_index][0][1]

    # if fingertip point lies below the hand contour center point:
    if finger_tip_y > cy:
        find_finger_tip = False
        for index in range(len(x)):
            x_p = int(x[index])
            y_p = int(y[index])
            if (x_p > cx -farthest_x_margin) and (x_p < cx + farthest_x_margin) and (y_p < cy):
                farthest_points.append((x_p, y_p))

        for j in range(3):
            closest_x = closest_x_margin + j * 4
            for i in range(len(farthest_points)):
                if (farthest_points[i][0] > cx - closest_x) and (farthest_points[i][0] < cx + closest_x):
                    finger_tip_x = farthest_points[i][0]  ;    finger_tip_y = farthest_points[i][1]
                    find_finger_tip = True
                    break
            if find_finger_tip:
                break

    finger_tip_x += int(detection_rec_x_start * frame_shape[1])
    finger_tip_y += int(detection_rec_y_start * frame_shape[0])
    finger_tip = (finger_tip_x,finger_tip_y)
    return finger_tip

def all_same(items):
     return all(x == items[0] for x in items)

def key_pressed(finger_tip_point):
    finger_tip_x = finger_tip_point[0]   ;  finger_tip_y = finger_tip_point[1]

    # check if the finger tip point is inside into any key rectangle:
    for i in range (len(key_rectangle_positions)):
        key_info = key_rectangle_positions[i]
        key_rec_x0 = key_info[0][0]  ;   key_rec_y0 = key_info[0][1]
        key_rec_x1 = key_info[1][0]  ;   key_rec_y1 = key_info[1][1]
        selected_key = key_info[2]
        if (finger_tip_x > key_rec_x0) and (finger_tip_x < key_rec_x1) and (finger_tip_y > key_rec_y0) and  (finger_tip_y < key_rec_y1):
            pressed_key_buffer.append(selected_key)
            return i

def draw_selected_key(frame,key_index):
    key_info = key_value_positions[key_index]
    key_rectangle_info = key_rectangle_positions[key_index]
    for i in range (12):
        for j in range (12):
           cv2.rectangle(frame, key_rectangle_info[0], key_rectangle_info[1], hover_rectangle_color, -1)
           if len(str(key_info[0])) > 1:
               cv2.putText(frame, key_info[2], (key_info[0], key_info[1]), font, 0.75, hover_color, 2)
           else:
               cv2.putText(frame, key_info[2], (key_info[0], key_info[1]), font, 1.5, hover_color, 3)

def mainProcess():
    _,frame=cap.read()
    frame = cv2.flip(frame, 1) # 1 for flipping around the y-axis
    output_image = frame.copy()
    output_image = cv2.bilateralFilter(output_image, 5, 50, 100) #smoothing filter
    global BG_captured
    global hand_hist_detected
    if BG_captured == False :
        # then the time hasn't reached yet its limit value so either 
        # the hand histogram hasn't been created yet or 
        # the background subtraction hasn't been performed
        global time_in_seconds
        if time_in_seconds < hand_hist_time_limit:
            var.set("Please position your hand on the green rectangle till it disappears\nPlease ONLY position your hand and move away your face\nDO NOT move your hand")
            x0, y0 = int(sample_hist_x[0]*frame.shape[0]), int(sample_hist_y[0]*frame.shape[1]) + (detection_rec_width // 2) -10 
            cv2.rectangle(output_image,(y0, x0),(y0 + 84,x0 + 116),hand_hist_rec_color, 1)
        elif time_in_seconds > hand_hist_time_limit and time_in_seconds < BG_sub_time_limit:
            text_label.place(x=700, y=300)
            var.set("Please move away from area inbounded in the blue rectangle\n In order to detect your surrounding Environment")
            cv2.rectangle(output_image,(detection_rec_x0,detection_rec_y0),(detection_rec_x1, detection_rec_y1),
                  detection_rec_color, 2)
        current_time = round(time.perf_counter())
        global previous_time
        if (current_time - previous_time) == 1:
            time_in_seconds +=1
            if time_in_seconds == hand_hist_time_limit :
                global hand_hist
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
                global fgbg
                fgbg = cv2.createBackgroundSubtractorMOG2(0,bg_sub_threshold) 
                BG_captured = True
            print(time_in_seconds)
        previous_time = current_time
        
    else:
        global input_word
        global password_entered
        global money_entered
        global button_action
        #global transaction_done
        global currentPage
        if currentPage == 1 :
            # Draw the keypad:
            draw_keypad(output_image)
            text_label.place(x=800, y=300)
            var.set("Please insert your Bank Card\nAnd Enter your Password")
            text_label.config(font=tkFont.Font(family="Lucida Grande", size=20 ))
            if input_word!='' :
                inputPass.set(input_word)
                text_label2.config(font=tkFont.Font(family="Lucida Grande", size=25 ))
                text_label2.place(x=900, y=500)      
            if password_entered :
                if len(input_word)==4 :
                    if input_word==password :
                        text_label2.place(x=800, y=400)
                        inputPass.set("Password entered Successfully")
                        input_word=''
                        currentPage=2
                    else:
                        text_label2.place(x=800, y=400)
                        inputPass.set("Wrong Password,Try again")
                        input_word=''
                        password_entered=FALSE
                else:
                    text_label2.place(x=800, y=400)
                    inputPass.set("Invalid Password Length\n Password should be 4 digits")
                    input_word=''
                    password_entered=FALSE
                    
        elif currentPage==2 :
            draw_deposit_withdraw_buttons(output_image)
            inputPass.set('')
            text_label.place(x=720, y=300)
            var.set("Please Choose a service of your desire")
            text_label.config(font=tkFont.Font(family="Lucida Grande", size=20 ))

            if button_action == 'Deposit' :
                currentPage=3
            if button_action == 'Withdraw' :
                currentPage=4
            if button_action == 'Inquiry' :
                currentPage=5

        elif currentPage == 3 :
            inputPass.set('')
            text_label.place(x=750, y=300)
            var.set("Please Insert only notes of 100,50,20\nNotes of 10 and 5 are not allowed")
            text_label.config(font=tkFont.Font(family="Lucida Grande", size=20 ))
            
            if deposit_done :
                text_label2.place(x=850, y=400)
                inputPass.set("Transaction Done\nHave a nice Day")
                input_word=''

        elif currentPage == 4 :
            # Draw the keypad:
            draw_keypad(output_image)
            text_label.place(x=750, y=300)
            var.set("Please enter the amount of money\nyou wish to withdraw")
            text_label.config(font=tkFont.Font(family="Lucida Grande", size=20 ))
            if(input_word!=''):
                inputPass.set(input_word+".00 EGP")
                text_label2.config(font=tkFont.Font(family="Lucida Grande", size=25 ))
                text_label2.place(x=900, y=500)
                
            if money_entered :
                if int(input_word) <= Balance :
                    text_label2.place(x=850, y=400)
                    inputPass.set("Transaction Done\nHave a nice Day")
                    input_word=''
                else:
                    text_label2.place(x=770, y=400)
                    text_label2.config(font=tkFont.Font(family="Lucida Grande", size=15 ))
                    inputPass.set("Exceeding Balance\n Please enter an amount within your balance\n "+str(Balance)+".00 EGP")
                    input_word=''
                    money_entered=FALSE
                        
        elif currentPage == 5 :
            inputPass.set('')
            text_label.place(x=850, y=300)
            var.set("Your current Balance is:\n"+str(Balance)+".00 EGP")
            text_label.config(font=tkFont.Font(family="Lucida Grande", size=20 ))
            
        # then the hand histogram is created and the background subtraction is performed
        roi = frame[detection_rec_y0:detection_rec_y0 + detection_rec_height,
                           detection_rec_x0:detection_rec_x0 + detection_rec_width]

        # create a mask of the hand using hand color histogram
        hist_mask = histMasking(roi, hand_hist)
        hist_mask = binarizeImage(hist_mask)
        #cv2.imshow("Histogram Mask",hist_mask)

        # create a mask of the hand using background subtraction
        mov_obj_mask = cropMovingObject(roi)
        mov_obj_mask = binarizeImage(mov_obj_mask)
        #cv2.imshow("Motion Mask",mov_obj_mask)

        # and the 2 masks to detect the moving hand
        hand_mask = cv2.bitwise_and(hist_mask, mov_obj_mask)
        #cv2.imshow("Hand Mask", hand_mask)

        # Find the contours of the hand mask:
        _,contours, hierarchy = cv2.findContours(hand_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)          # hand palm is the largest contour area.
            center = contour_centroid(max_contour)                    # Find the center of the hand palm.
            hull = cv2.convexHull(max_contour, returnPoints=False)    # Find the convex hull for the hand palm:
            convex_defects = cv2.convexityDefects(max_contour, hull)
            if convex_defects is not None:
                defects_points = convex_defects[:, 0][:, 0]           # Convex hull start points
                # Find the fingertip point of the hand palm using the convex hull:
                fingertip_point = get_fingertip(defects_points,max_contour,center,output_image.shape)
                cv2.circle(output_image, fingertip_point, 20, hover_circle_color, 3)
                cv2.line(output_image, (fingertip_point[0] - 50, fingertip_point[1]), (fingertip_point[0] + 50, fingertip_point[1]), hover_line_color, 3)
                cv2.line(output_image, (fingertip_point[0], fingertip_point[1] - 50), (fingertip_point[0], fingertip_point[1] + 50), hover_line_color, 3)

                # find the key pressed and write it:
                key_index = key_pressed(fingertip_point)
                # if a key has been selected for 10 frames, write the key and clear the buffer
                if len(pressed_key_buffer) > 15 :
                   list = pressed_key_buffer[5:]
                   if all_same(list):              # check if all items inside the list are identical
                       if len(list[0]) == 1:
                           input_word += list[0]
                       else:
                           button_action = list[0]
                       draw_selected_key(output_image,key_index)
                   pressed_key_buffer.clear()

    #cv2.imshow('V-PAD',output_image)
    tk_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)
    tk_output_image = Image.fromarray(tk_output_image)
    tk_output_image = ImageTk.PhotoImage(tk_output_image)
    video_stream_label.imgtk = tk_output_image
    video_stream_label.configure(image = tk_output_image)
    video_stream_label.after(1, mainProcess)



# create Tkinter window
root = Tk()
root.wm_title("V_PAD")
root.geometry("1280x800")

background_image = PhotoImage(file="/home/mohned/Desktop/V-PAD/hand.png")
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
root.update()
root.after(welPage_delay,)
background_label.pack_forget()


background_image = PhotoImage(file="/home/mohned/Desktop/V-PAD/img2.png")
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

video_stream_label = Label(background_label)
video_stream_label.place(x=30, y=100)


var = StringVar()
text_label = Label(background_label, textvariable=var, font = tkFont.Font(family="Lucida Grande", size=15 ))
text_label.place(x=690, y=300)

inputPass = StringVar()
text_label2 = Label(background_label, textvariable=inputPass, font = tkFont.Font(family="Lucida Grande", size=15 ))

#capture webcam video  
cap = cv2.VideoCapture(0) # object for the video handle
cap.set(3, 1920) # change width to 1920 pixels
cap.set(4, 1080) #change height to 1080 pixels
cap.set(10, 200) #change brightness to 200
_,frame=cap.read()
detection_rec_x0 = int(detection_rec_x_start * frame.shape[1]) # The top left point x value                     
detection_rec_y0 = int(detection_rec_y_start * frame.shape[0]) # The top left point y value
detection_rec_x1 = int(detection_rec_x_end * frame.shape[1])   # The bottom right point x value
detection_rec_y1 = int(detection_rec_y_end * frame.shape[0])   # The bottom right point y value
detection_rec_height = detection_rec_y1 - detection_rec_y0
detection_rec_width = detection_rec_x1 - detection_rec_x0

#main loop
mainProcess()

root.mainloop()

   
