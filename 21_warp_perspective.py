import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('img/poker.jpg')

plt.imshow(img)


width, height = 250, 350

pts1 = np.float32([[817, 110], [949, 225], [571, 214], [696, 338]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

matrix = cv.getPerspectiveTransform(pts1, pts2)
imgOutput = cv.warpPerspective(img, matrix, (width, height))

cv.imshow('Output', imgOutput)
plt.show()
cv.waitKey(0)






