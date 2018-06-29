import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import math
import histogram as h
import cumulative_histogram as ch
img=cv2.imread('image.png',cv.IMREAD_GRAYSCALE)
height=img.shape[0]
width=img.shape[1]
pixels=width*height
hist=h.histogram(img)
cum_hist=ch.cumlatve_histogram(hist)
for i in np.arrange(height):
  for j in np.arrange(width): 
	a= img.item(i,j)
	b=math.floor(cum_hist[a]*255.0/pixels)
	img.itemset((i,j),b)

cv2.imwrite('hist.png',img)
cv2.imshow('image',img)
cv2.waitkey(0)





