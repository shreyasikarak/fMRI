import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import pdb

'''import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("image", required = True,
	help = "Path to the image to be thresholded")
parser.add_argument("threshold", type = int, default = 128,
	help = "Threshold value")
args = vars(parser.parse_args())'''

path = '/home/silp150/shreyashi/100307/unprocessed/3T/tfMRI_EMOTION_RL'
os.chdir(path)
threshold = 100

image = cv2.imread('image.png')
yrow, xcol, depth = image.shape
gray = np.zeros((yrow, xcol))

'''cnt_row = 0
for row in image:	
	cnt_col = 0
	for col in row:
		gray[cnt_row, cnt_col] = np.sum(col)/3		
		cnt_col = cnt_col + 1
	cnt_row = cnt_row + 1'''


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
# plt.show()

methods = [
	("THRESH_BINARY", cv2.THRESH_BINARY),
	("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
	("THRESH_TRUNC", cv2.THRESH_TRUNC),
	("THRESH_TOZERO", cv2.THRESH_TOZERO),
	("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]
 
for (threshName, threshMethod) in methods:
	
	(T, thresh) = cv2.threshold(gray, threshold, 255, threshMethod)

	plt.imshow(thresh, cmap='gray')
	plt.show()
