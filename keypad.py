from tkinter import*
import numpy as np
import cv2
import math
#import tkinter as tk
from PIL import Image
from PIL import ImageTk

#Set up GUI
window = Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = Frame(window, width=1920, height=1080)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)
width, height = 1920, 1080
cap.set(3, width)
cap.set(4, height)

# create a method that can separate the foreground from the background
fgbg = cv2.createBackgroundSubtractorMOG2()

# define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (13, 32, 210)

letter_selected = False

# create a list for the word
word = ''

# create a letter buffer
letter_buffer = []

letter_positions = []

# define frame_num
frame_num = 0

# Function that determines distance between two points
def distance(p0,p1):
    return math.sqrt((p1[1]-p0[1])**2+(p1[0]-p0[0])**2)



# create an overlay image. You can use any image
foreground = np.ones((80,90,3),dtype='uint8')*255
# Set initial value of weights
alpha = 0.2
# main loop:
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    added_image = cv2.addWeighted(frame[0:80,0:90,:],alpha,foreground[0:80,0:90,:],1-alpha,0)
    # Change the region with the result
    frame[0:80,0:90] = added_image
    # For displaying current value of alpha(weights)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'alpha:{}'.format(alpha),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    

    # 2.create a composite image that includes the webcam
    composite = frame.copy()

    # 3.define number position
    x_position = 250
    y_position = 70
    dx = 0
    dy = 0

    # 4.define hand position
    hand_position = 0, 0
    hand_on_keyboard = False
    number_selected = False

    # 5.Draw the Keypad numbers:
    Keypad_number = 1
    cv2.putText(composite, "Keypad", (160,80), font, 2, (255,0,0), 3)

    for row_number in range(3):
        x_position = 550
        y_position = y_position + dy
        for col_number in range(3):
            # print("letter value", letter)
            x_position = x_position + dx
            xy = x_position, y_position
            dx = 140
            cv2.putText(composite,str(Keypad_number), xy, font, 2, color, 2)
            letter_positions.append((str(Keypad_number), xy))
            if letter_selected:
                cv2.putText(composite, closest_letter, close_letter_position, font, 2, (255, 0, 0), 3)
            # number_positions.append((chr(40 + col_number), xy))
            Keypad_number += 1
        dx = 0
        dy = 90

    # add a line to show where the keyboard starts
    cv2.line(composite, (composite.shape[1], 270), (0, 270), (0,255,0), 2)

    look = frame[50:280,  0:frame.shape[1]]
    fgmask = fgbg.apply(look)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 



#Slider window (slider controls stage position)
sliderFrame = Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


show_frame()  #Display 2
window.mainloop()  #Starts GUI
