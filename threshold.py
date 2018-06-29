import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img= cv.imread('image.png')
retval, threshold=cv.threshold(img,12,255,cv.THRESH_BINARY)
grayscaled= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
retval2,threshold2=cv.threshold(grayscaled,12,255,cv.THRESH_BINARY)
gaus=cv.adaptiveThreshold(grayscaled,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,115,1)
cv.imshow('original',img)
cv.imshow('threshold',threshold)
cv.imshow('threshold2',threshold2)
