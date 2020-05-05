import cv2
import numpy as np
import math

adaptive_threshold_max_value = 100
Kernel_size = (3,3)
contour_color = (0,255,0)
center_point_color = (23,208,253)
farthest_point_color = (255, 0, 102)
convex_hull_color = (0,0,255)

# Read a created hand video:
cap = cv2.VideoCapture('hand.avi')
if cap.isOpened() == False:
    print("Error during reading the video")
    exit()

while True:
    # 1.Read each video frame:
    ret,frame = cap.read()
    composite = frame.copy()

    # 2.Apply an image pre-processing:
    # 2.1.Convert each frame to gray scale then make an adaptive threshold:
    gray_image = cv2.cvtColor(composite,cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray_image, 170, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    _, thresh_image = cv2.threshold(thresh_image, 10, 255, cv2.THRESH_BINARY_INV)

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
    cv2.circle(frame,center,5,center_point_color,3)

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

        # 6.Calculate the farthest point from the center of the palm hand(Finger tip):
        s = convex_defects[:, 0][:, 0]
        x = np.array(max_contour[s][:, 0][:, 0], dtype=np.int)
        y = np.array(max_contour[s][:, 0][:, 1], dtype=np.int)

        farthest_points = []
        find_points = False
        for index in range (len(x)):
            if (x[index] > cx-100) and (x[index] < cx+100) and (y[index] < cy):
                find_points = True
                farthest_points.append((x[index],y[index]))
                break
        print('center = ',center,'farthest_points = ',farthest_points)

        if find_points:
             if len(farthest_points) == 1:
                 cv2.circle(frame,farthest_points[0],10,farthest_point_color,4)


    cv2.imshow('hand_detection frame', frame)
    key = cv2.waitKey(1500) & 0xff
    if key == 27:  ## if the key is esc character break the loop then close the video streaming
        break

cap.release()
cv2.destroyAllWindows()