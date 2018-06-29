import cv2
srcimg=cv2.imread("image.png")
srcimage=cv2.resize(srcimg,(320,240))
blurimg=cv2.GaussianBlur(srcimage,(5,5),3)
dstimg=cv2.Canny(blurimg,500,200)
cv2.imshow('org',srcimage)
cv2.imshow('Canny', dstimg)
cv2.waitKey(0)


