import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('img/shanghai.jpg')
cv.imshow('Shanghai', img)

# BGR is the default color space used by openCV (blue, green, red) instead of RGB
'''
plt.imshow(img)
plt.show()
'''
# matplotlib uses the RBG color space so the shown image will be different

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV (Hue Saturation Value, based on how humans think and conceive color)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*a*b (based on how humans think and conceive color)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('L*a*b', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)
plt.imshow(rgb)
plt.show()

cv.waitKey(0)
