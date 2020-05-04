import cv2
import glob

# 1.Define a list to include all video images:
img_array = []
height ,width = 703,800
size = (width,height)

# 2.read all hand images from its directory then store at the list after resizing them:
for filename in glob.glob('images\*.jpg'):
    img = cv2.imread(filename)
    img = cv2.resize(img,size)
    img_array.append(img)

# 3.Create a video with frame rate (0.5 fps):
out = cv2.VideoWriter('hand.avi',cv2.VideoWriter_fourcc(*'DIVX'),1,size)

# 4.write all hand images from the list into the video:
for i in range (len(img_array)):
    out.write(img_array[i])

cv2.waitKey(500)
out.release()


