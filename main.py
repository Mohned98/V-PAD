from tkinter import *
import keypad
import tkinter.font as tkFont
import tkinter.messagebox as tkMessageBox


root = Tk()
root.wm_title("V_Keypad")

# set window size
root.geometry("800x500")

#set window color
root.config(bg="#900C3F")

#set font style
fontStyle = tkFont.Font(family="Lucida Grande", size=30 )

welcome = Label(root ,text="welcome to our V_Keypad" ,bg="#900C3F" ,fg="white"  ,font=fontStyle)
welcome.pack( pady=10,fill=X)


def start_btn():
    welcome.pack_forget()
    Startbutton.pack_forget()
    res=keypad()



Startbutton = Button( root, text="Let's Start...", fg="blue", command=start_btn ,font = fontStyle)

#set position of button
Startbutton.place(x=250,y=170)



root.mainloop()

