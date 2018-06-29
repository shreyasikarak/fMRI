import cv2
import numpy
import matplotlib.pyplot as plt
import pdb
 
img = cv2.imread('image.png')
img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
#img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

cv2.imwrite('result.jpg',hist_equalization_result)

im=cv2.imread('image.png')
im=im/255.0
im_power_law_transformation=cv2.pow(im,0.6)
plt.imshow(im)
plt.imshow(hist_equalization_result)
plt.imshow(im_power_law_transformation)
plt.show()
#pdb.set_trace()
cv2.waitkey(0)
