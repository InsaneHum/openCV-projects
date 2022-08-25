import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('img/car.jpg')
# cv.imshow('Shanghai', img)

# create mask
blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 300, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)

# histograms show the pixel density of intensities
# color histogram
colors = ('b', 'g', 'r')

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.xlim([0, 256])

for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)

plt.show()

cv.waitKey(0)

