import cv2 as cv

img = cv.imread('img/sinfeng.JPG')
img = cv.resize(img, (1000, 1000), interpolation=cv.INTER_AREA)
cv.imshow('Pepe', img)


# converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert color (RGB TO GRAY)

cv.imshow('Grayscale', gray)


# blur
blur = cv.GaussianBlur(img, (15, 15), cv.BORDER_DEFAULT)
# kernel size used to compute the blur of the image, must be odd

cv.imshow('Blur', blur)


# edge cascade (find edges that are present in the image)
canny = cv.Canny(img, 125, 175)

cv.imshow('Canny', canny)


# dilating the image (enhance edge features)
dilated = cv.dilate(canny, (7, 7), iterations=3)
# kernel also used here

cv.imshow('Dilated', dilated)


# eroding the edges
eroded = cv.erode(dilated, (7, 7), iterations=1)

cv.imshow('Eroded', eroded)


# resize (ignores aspect ratio)
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)  # cv.INTER_AREA for shrinking
# use cv.INTER_LINEAR or cv.INTER_CUBIC for enlarging

cv.imshow('Resized', resized)


# cropping (array slicing)
cropped = img[50:200, 200:400]

cv.imshow('Cropped', cropped)

cv.waitKey(0)
