import cv2 as cv
import numpy as np

img = cv.imread('img/car.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)  # computes the gradients of the grayscale image
lap = np.uint8(np.absolute(lap))  # compute the absolute values of the gradients as pixels cannot have negative values

cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)  # (image, depth, direction (1, 0 for x))
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)  # (0, 1 for y)
combined_sobel = cv.bitwise_or(sobelx, sobely)

# cv.imshow('Sobel X', sobelx)
# cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

# Canny (for comparison)
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)
