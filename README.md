# V-PAD
## Virtual Keypad

## Team Members:
| Name                            | ID          |
| :----:                          |    :----:   |
| Mohand Gamal Fawzy Helmy        | 1501516     |
| Mohned Mohamed Abd El-Hamied    | 1501519     |
| Yasmeen Mostafa Hassan Elnaggar | 1501717     |
| Yasmine Mohamed Mohamed Elsayed | 1501714     |


### Project Description:
The project is basically a virtual keypad to provide a seamless and efficient alternative control for any touch screen, provided that, there is a camera. This will significantly help to minimize the spread of viruses and germs residing on commonly used surfaces that is open for the public. As we use touch many surfaces to get some normal day-to-day services, for example using the ATM, buying from a vending machine and ordering food with touch screens at some restaurants. This can also help with the current outbreak of the coronavirus (COVID-19) and maybe lessen the chances of any future outbreaks.  
 
As a prototype for the project, we will mainly focus on implementing an ATM-similar interface using the laptop webcam to prove the applicability of the project’s idea. A project that can later be installed on every ATM machine without the need for any additional hardware set up, as all ATM machines come with a preinstalled camera of their own.

### Implementation Details: 
#### User Interface: 
- The project aims to provide a new and revolutionary user interface that is seamless to use while at the same time provides a hygienic standard that was mistakenly overlooked in today’s user interfaces.
- We used Tkinter Python framework to implememt the GUI part of our project
#### Algorithms used:
The implementation would mainly use image processing technologies like OpenCV library to detect the hand and finger movement by mainly 2 steps:
# HSV Segmentation 
 In HSV (Hue, Saturation, Value) segmentation, the idea is to segment the hand based on the color. At first, we will sample the color of the hand. The reason for using HSV rather than RGB to eliminate the brightness because this is an issue when we detect the hand because the hand has to be under the same brightness in order to be detected. The brightness of a color is encoded in the Value (V) in the HSV. Hence, when we sample the color of the hand, we sample only the Hue (H) and Saturation (S)
 
'handHist = cv2.calcHist([ROI], [0, 1], None, [180, 256], [0, 180, 0, 256])'
  
# Background subtraction
we need to have a background image (without the hand) first. To find the hand, we can subtract the image with hand from the background.

'fgbg = cv2.createBackgroundSubtractorMOG2(0,bg_sub_threshold)'

then subtract the background form the frame for each input one to extract the hand mask
#### Outputs and Inputs: 
The GUI will output for the user a keypad with the needed functionalities on the screen and a live video of the camera’s feed at any time. The user will use his/her hand and finger gestures to interact with the keypad. Specific detected gestures will be considered as inputs for the system. 
