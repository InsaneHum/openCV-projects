import cv2 as cv
import numpy as np

# process image
img = cv.imread('img/J20.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale', gray)
blur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)

# find edges using canny
canny = cv.Canny(blur, 100, 175)  # (image, intensity lower limit, intensity upper limit)
# canny_inv = cv.bitwise_not(canny)
# cv.imshow('Canny', canny)

# find edges using threshold
ret, thresh = cv.threshold(blur, 125, 255, cv.THRESH_BINARY)  # threshold binarizes the image (set to 0 if below lower limit, set to 1 if above upper limit), type
# cv.imshow('Thresh', thresh)

# contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# contours is a list that contains all found contours in the image
# hierarchies refers to the hierarchical representation of contours (etc square in rectangle, circle in square...)
# cv.RETR_... is a mod in which the .findContours method finds contours,
# _LIST returns all contours, _EXTERNAL only returns external contours, _TREE returns all hierarchical contours
# contour approximation method (how we approximate the contours):
# cv.CHAIN_APPROX_NONE does nothing, just returns all contours; cv.CHAIN_APPROX_SIMPLE, compresses all the contours into a simple one that makes most sense

print(f'{len(contours)} contour(s) found')

# visualize contours
blank = np.zeros(img.shape, dtype='uint8')  # create blank image with same size as og image
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)  # canvas to draw on, list of contours, how many contours to draw (-1 for all), color, thickness)
cv.imshow('Contours', blank)

cv.waitKey(0)
