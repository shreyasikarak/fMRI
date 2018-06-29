import cv2
import numpy as np
import nilearn
import argparse
import pdb
import matplotlib.pyplot as plt

im=cv2.imread('image.png')
im=im/255.0
im_power_law_transformation=3*cv2.pow(im,0.0)
plt.imshow(im)
plt.imshow(im_power_law_transformation)

plt.show()
#pdb.set_trace()
cv2.waitkey(0)
