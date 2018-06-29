import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('image.png')
blurred= cv2.pyrMeanShiftFiltering(image, 31,91)
gray= cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret , threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY* cv2.THRESH_OTSU)
_, contours,_= cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image,contours,0,(0,0,255),0)
#cv2.namedWindow('Display',cv2.WINDOW_NORMAL)

plt.imshow(image)
plt.show()

#cv2.waitKey()
