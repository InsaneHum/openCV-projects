import cv2 as cv
import numpy as np

img = cv.imread('img/hollowknight.jpg')

cv.imshow('The Knight', img)


# translation
def translate(image, x, y):  # shifts the image by x, y (pixels)
    # -x --> Left
    # -y --> Up
    # x --> Right
    # y --> Down
    transMat = np.float32([[1, 0, x], [0, 1, y]])  # create empty matrix
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)


translated = translate(img, -300, 100)
cv.imshow('Translated', translated)


# rotation
def rotate(img, angle, rotPoint=None):  # rotates the image by angle and a rotation point
    # +angle --> counter clockwise
    # -angle --> clockwise
    (height, width) = img.shape[:2]

    if rotPoint is None:  # if no rotation point is specified, set rotation point as center of image
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)  # set scale = 1
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)


# flipping
flipped = cv.flip(img, 1)
# 0 --> flip vertically
# 1 --> flip horizontally
# -1 --> flip both vertically and horizontally
cv.imshow('Flipped', flipped)


cv.waitKey(0)
