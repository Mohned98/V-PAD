from tkinter import *
from PIL import ImageTk, Image
import tkinter.font as tkFont
import tkinter.messagebox as tkMessageBox
import numpy as np
import cv2
import math
import subprocess


root = Tk()
root.geometry("1000x500")

# Create a frame
left = Frame(root, borderwidth=2, relief="solid")
right = Frame(root, borderwidth=2, relief="solid")
container1 = Frame(left,  borderwidth=2,width = 600,height = 500 , relief="solid" )
container2 = Frame(right, borderwidth=2,width = 500,height = 400 , relief="solid",bg="#900C3F")
container1.grid()

# Create a label in the frame
lmain = Label(container1)
lmain.grid()
#lmain.pack(fill=BOTH, expand=YES)


#set font style
fontStyle = tkFont.Font(family="Lucida Grande", size=15 )
password = Label(container2, text="please enter your password",bg="#900C3F" ,fg="white"  ,font=fontStyle)

left.pack(side="left", expand=True, fill="both")
right.pack(side="right", expand=True, fill="both")
container1.pack(expand=True, fill="both", padx=5, pady=5)
container2.pack(expand=True, fill="both", padx=5, pady=5)
password.pack(side="left")
password.place(x=5,y=5)


# Capture from camera
cap = cv2.VideoCapture(0)
width, height = 600,480
cap.set(3, width)
cap.set(4, height)

# create a method that can separate the foreground from the background
fgbg = cv2.createBackgroundSubtractorMOG2()


# define alphabet position
position = 0
# define hand position
hand_position = 0, 0
hand_on_keyboard = False
letter_selected = False
# define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (13, 32, 210)

# create a list for the word
word = ''

# create a letter buffer
letter_buffer = []

# define frame_num
frame_num = 0


# function that determines distance between two points
def distance(p0, p1):
    return math.sqrt((p1[1] - p0[1]) ** 2 + (p1[0] - p0[0]) ** 2)

# function for video streaming
def video_stream():
    _, frame = cap.read()
     # create a composite image that includes the webcam
    composite = frame.copy()

    # add the letters
    # make a list of letter positions
    letter_positions = []
    for letter in range(150):
        x_position = position + letter * 200
        y_position = 150
        xy = x_position, y_position
        cv2.putText(composite, chr(40 + letter), xy, font, 2, color, 3)
        letter_positions.append((chr(40 + letter), xy))
        # if there is a letter selected, make that letter green
        if letter_selected:
            cv2.putText(composite, closest_letter, close_letter_position, font, 2, (255, 0, 0), 3)
    # add a line to show where the keyboard starts
    cv2.line(composite, (composite.shape[1], 200), (0, 200), color, 2)


    # find the background
    # look only at the keyboard part of the frame
    look = frame[50:200, 0:frame.shape[1]]
    fgmask = fgbg.apply(look)
    #cv2.imshow('fgmask',fgmask)
      

    cv2image = cv2.cvtColor(composite, cv2.COLOR_BGR2RGBA)
    
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 



video_stream()
root.mainloop()
