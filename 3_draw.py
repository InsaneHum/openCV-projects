import cv2 as cv
import numpy as np

# create blank image
blank = np.zeros((500, 500, 3), dtype='uint8')  # create a zero matrix of size 500*500 and a colour channel of 3 (RGB)

cv.imshow('Blank', blank)

# 1. paint the image to a certain colour
blank[:] = 0, 255, 0  # set all pixels in the blank array to a certain color

cv.imshow('Green', blank)

# 2. paint a certain part of the image to a certain color
blank[200:300, 300:400] = 0, 0, 255  # set pixels in the range to a certain color

cv.imshow('Red square', blank)

# 3. draw a rectangle
blank = np.zeros((500, 500, 3), dtype='uint8')  # reset blank canvas
cv.rectangle(blank, (0, 0), (120, 300), (0, 255, 0), thickness=3)
# (canvas to draw on, start pt coords, end pt coords, rgb color, thickness)

cv.imshow('Rectangle', blank)

# 4. draw a rectangle scaled to size
blank = np.zeros((500, 500, 3), dtype='uint8')  # reset blank canvas
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=-1)
# set thickness to -1 for color fill

cv.imshow('Scaled Rectangle', blank)

# 5. draw a circle
blank = np.zeros((500, 500, 3), dtype='uint8')  # reset blank canvas
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=3)
# (canvas to draw on, coords of center of circle, radius, rgb color, thickness)

cv.imshow('Circle', blank)

# 6. draw a line
blank = np.zeros((500, 500, 3), dtype='uint8')  # reset blank canvas
cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)
# (canvas to draw on, start pt coords, end pt coords, rgb color, thickness)

cv.imshow('Line', blank)

# 7. write text
blank = np.zeros((500, 500, 3), dtype='uint8')  # reset blank canvas
cv.putText(blank, 'beans', (225, 225), cv.FONT_HERSHEY_TRIPLEX, 3.0, (255, 255, 255), thickness=2)
# (canvas to draw on, 'text', start pt coords, font, scale, rgb color, thickness)

cv.imshow('Text', blank)

cv.waitKey(0)
