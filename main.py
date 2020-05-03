from tkinter import *
import tkinter.font as tkFont
import tkinter.messagebox as tkMessageBox



root = Tk()

#set window title
root.wm_title("V_Keypad")

# set window size
root.geometry("800x400")


# set a  welcome label
welcome = Label(root ,text="welcome to our V_Keypad" ,fg="black")
welcome.pack()



def start_btn():
    tkMessageBox.showinfo( "Hello", "start")

    

# create a start button
Startbutton = Button( root, text="Let's Start...", command=start_btn , fg="black")
Startbutton.pack()


root.mainloop()

