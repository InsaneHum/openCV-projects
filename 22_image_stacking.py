import numpy as np
import cv2 as cv

img = cv.imread('img/poker.jpg')


def stackImages(scale, image, size):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)

    image = cv.resize(image, dimensions, interpolation=cv.INTER_AREA)
    output = image

    for col in range(size[1] - 1):
        output = np.hstack((output, image))

    vertical_output = output

    for row in range(size[0] - 1):
        vertical_output = np.vstack((vertical_output, output))

    return vertical_output


'''
# horizontal stack
hor = np.hstack((img, img))

cv.imshow('Horizontal Stack', hor)

# vertical stack
hor = np.vstack((img, img))

cv.imshow('Vertical Stack', hor)
'''

cv.imshow('Stacked', stackImages(0.3, img, (2, 3)))
cv.waitKey(0)
