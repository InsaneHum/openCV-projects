import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('img/shanghai.jpg')
# cv.imshow('Shanghai', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# create mask
blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 300, 255, -1)
# cv.imshow('Mask', mask)

# histograms show the pixel density of intensities
# grayscale histogram
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])  # (images, color channels, mask, histogram_size, range)

# plotting
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

cv.waitKey(0)
