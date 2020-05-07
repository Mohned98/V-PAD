from tkinter import *
from PIL import ImageTk, Image
import cv2


root = Tk()
root.geometry("1200x600")
# Create a frame
#app = Frame(root, bg="white")
left = Frame(root, borderwidth=2, relief="solid")
#left.winfo_geometry("850x600")
right = Frame(root, borderwidth=2, relief="solid")
container1 = Frame(left,  borderwidth=2,width = 500,height = 400 , relief="solid")
container2 = Frame(right, borderwidth=2, relief="solid")




container1.grid()
# Create a label in the frame
lmain = Label(container1)
lmain.grid()

label1 = Label(container2, text="please enter your password",bg="#900C3F" ,fg="white"  ,font=fontStyle)

left.pack(side="left", expand=True, fill="both")
right.pack(side="right", expand=True, fill="both")
container1.pack(expand=True, fill="both", padx=5, pady=5)
container2.pack(expand=True, fill="both", padx=5, pady=5)
label1.pack()

# Capture from camera
cap = cv2.VideoCapture(0)
width, height =  850, 600
cap.set(3, width)
cap.set(4, height)
# function for video streaming
def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

video_stream()
root.mainloop()
