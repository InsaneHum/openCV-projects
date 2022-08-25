import numpy as np
import cv2 as cv

myColors = []


def resizeImage(scale, image):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)


img = resizeImage(0.8, cv.imread('img/car.jpg'))

# convert image to HSV to detect color
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


# create function that changes the values everytime the trackbar is altered
def empty(a):
    pass


# use trackbars to find target HSV values
cv.namedWindow('TrackBars')
cv.resizeWindow('TrackBars', 640, 250)
cv.createTrackbar('Hue Min', 'TrackBars', 0, 180, empty)
cv.createTrackbar('Hue Max', 'TrackBars', 18, 180, empty)
cv.createTrackbar('Sat Min', 'TrackBars', 48, 255, empty)
cv.createTrackbar('Sat Max', 'TrackBars', 222, 255, empty)
cv.createTrackbar('Val Min', 'TrackBars', 78, 255, empty)
cv.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)

# grab values from track bars
while True:
    h_min = cv.getTrackbarPos('Hue Min', 'TrackBars')
    h_max = cv.getTrackbarPos('Hue Max', 'TrackBars')
    s_min = cv.getTrackbarPos('Sat Min', 'TrackBars')
    s_max = cv.getTrackbarPos('Sat Max', 'TrackBars')
    v_min = cv.getTrackbarPos('Val Min', 'TrackBars')
    v_max = cv.getTrackbarPos('Val Max', 'TrackBars')

    # print(h_min, h_max, s_min, s_max, v_min, v_max)

    # filter the images based on the HSV values
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    
    imgResult = cv.bitwise_and(img, img, mask=mask)

    stacked = np.hstack((np.vstack((img, imgResult)), np.vstack((hsv, np.dstack((mask, mask, mask))))))

    cv.imshow('Stacked', stacked)
    cv.waitKey(1)
