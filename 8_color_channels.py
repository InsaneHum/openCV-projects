import cv2 as cv
import numpy as np

img = cv.imread('img/shanghai.jpg')
cv.imshow('Shanghai', img)

blank = np.zeros(img.shape[:2], dtype='uint8')  # data type 'uint8' is for images

b, g, r = cv.split(img)  # splits the image into its rgb components in pixel concentrations

# restructure and show the images only by their rgb components
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(f'OG: {img.shape}')
print(f'Blue: {b.shape}')
print(f'Green: {g.shape}')
print(f'Red: {r.shape}')

merged = cv.merge([b, g, r])  # merges the image back from its components
# cv.imshow('Merged', merged)

cv.waitKey(0)
