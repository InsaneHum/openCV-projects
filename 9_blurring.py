import cv2 as cv
import numpy as np

img = cv.imread('img/shanghai.jpg')
cv.imshow('Shanghai', img)

# averaging
average = cv.blur(img, (5, 5))
cv.imshow('Average Blur', average)

# gaussian blur (more natural
gauss = cv.GaussianBlur(img, (5, 5), 0)  # sigmaX (standard deviation in x axis)
cv.imshow('Gaussian Blur', gauss)

# median blur (more effective in reducing noise)
median = cv.medianBlur(img, 5)  # no tuple needed
cv.imshow('Median Blur', median)

# bilateral (retains the edges of the image)
bilateral = cv.bilateralFilter(img, 10, 35, 25)
# (image, radius, sigmaColor(how many colors around the pixel that will be considered when the blur is computed),
# spaceSigma(how far pixels away from the central pixel will influence the blurring)
cv.imshow('Bilateral Filtering', bilateral)

cv.waitKey(0)
