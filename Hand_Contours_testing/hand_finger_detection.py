import cv2
import numpy as np
import math

adaptive_threshold_max_value = 100
Kernel_size = (3,3)
contour_color = (0,255,0)
center_point_color = (23,208,253)
enclosing_circle_color = (255,0,0)
farthest_point_color = (255, 0, 102)
convex_hull_color = (0,0,255)
finger_tip = (0,0)
dx = 4

height ,width = 500,500
size = (width,height)

# Read a created hand video:
cap = cv2.VideoCapture('hand.avi')
if cap.isOpened() == False:
    print("Error during reading the video")
    exit()

#write the video:
out = cv2.VideoWriter('hand_detection.avi',cv2.VideoWriter_fourcc(*'DIVX'),1,size)

while True:
    # 1.Read each video frame:
    ret,frame = cap.read()
    composite = frame.copy()

    # 2.Apply an image pre-processing:
    # 2.1.Convert each frame to gray scale then make an adaptive threshold:
    gray_image = cv2.cvtColor(composite,cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # 2.2.Make an Open Operation to remove the white noise:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Kernel_size)
    detection_mask = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel,iterations=1)

    # 3.Find the contours of the largest area (hand Palm):
    contours , hierarchy = cv2.findContours(detection_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(frame,max_contour,-1,contour_color,3)

    # 4.Find the center of the contour:
    moments = cv2.moments(max_contour)
    if moments['m00'] != 0:
      cx = int(moments['m10'] / moments['m00'])
      cy = int(moments['m01'] / moments['m00'])
    center = (cx,cy)
    cv2.circle(frame,center,3,center_point_color,3)

    # 5.Find the finger tip using Convex hull:
    hull = cv2.convexHull(max_contour, returnPoints=False)
    convex_defects = cv2.convexityDefects(max_contour, hull)
    if convex_defects is not None:
        for i in range(convex_defects.shape[0]):
            s, e, f, d = convex_defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            cv2.line(frame, start, end, convex_hull_color, 2)

        # 6.Calculate the finger tip from the center of the palm hand(Finger tip):
        s = convex_defects[:, 0][:, 0]
        x = np.array(max_contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(max_contour[s][:, 0][:, 1], dtype=np.float)

        Xpoints_subtract_Xcenter = cv2.pow(cv2.subtract(x, cx), 2)
        Ypoints_subtract_Ycenter = cv2.pow(cv2.subtract(y, cy), 2)
        distance = cv2.sqrt(cv2.add(Xpoints_subtract_Xcenter, Ypoints_subtract_Ycenter))
        max_distance_index = np.argmax(distance)
        if max_distance_index < len(s):
            finger_tip = s[max_distance_index]
            finger_tip = tuple(max_contour[finger_tip][0])

        # farthest point is under the center point:
        if finger_tip[1] > cy:
           farthest_points = []
           find_finger_tip = False
           for index in range (len(x)):
               x_p = int(x[index])
               y_p = int(y[index])
               if (x_p > cx-80) and (x_p < cx+80) and (y_p < cy):
                  farthest_points.append((x_p,y_p))

           for j in range (3):
               closest_x = dx + j * 4
               for i in range (len(farthest_points)):
                  if (farthest_points[i][0] > cx - closest_x) and (farthest_points[i][0] < cx + closest_x):
                     finger_tip = farthest_points[i]
                     find_finger_tip = True
                     break

               if find_finger_tip:
                   break

        cv2.circle(frame, finger_tip, 3, farthest_point_color, 3)

    cv2.imshow('hand_detection frame', frame)
    out.write(frame)
    key = cv2.waitKey(1500) & 0xff
    if key == 27:  ## if the key is esc character break the loop then close the video streaming
        break

cap.release()
out.release()
cv2.destroyAllWindows()