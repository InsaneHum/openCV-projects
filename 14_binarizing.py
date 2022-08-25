import cv2 as cv

img = cv.imread('img/J20.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# simple thresholding
threshold, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# if pixel intensity is larger than the thresholding value(150), set pixel intensity to maxval(255), else set pixel intensity to 0
# thresh = thresholded(binarized) image
# threshold = thresholding value
# cv.imshow('Simple Thresholding', thresh)

# inversed simple thresholding
threshold_inv, thresh_inv = cv.threshold(gray, 125, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simple Thresholding Inversed', thresh_inv)

# adaptive thresholding (the computer finds the optimal threshold value by itself)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 9)
# (image, maxval, adaption method (method that computes the optimal threshold value),
# threshold type, kernel size, c value (integer that is subtracted from the mean to find tune threshold)

cv.imshow('Adaptive Thresholding', adaptive_thresh)


cv.waitKey(0)
