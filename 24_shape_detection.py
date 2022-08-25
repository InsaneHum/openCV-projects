import numpy as np
import cv2 as cv

img = cv.imread('img/shapes.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (7, 7), 1)

canny = cv.Canny(imgBlur, 50, 50)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None,
                                               scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


imgContour = img.copy()
blank = np.zeros_like(img, dtype='uint8')


def getContours(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:
            cv.drawContours(imgContour, cnt, -1, (0, 0, 0), 6)
            # canvas, contour, how many contours (-1 for all), color, thickness

            # calculate arclength to find corners (contour, is shape closed)
            peri = cv.arcLength(cnt, True)

            # approx the corner points of the shape
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            objCor = len(approx)

            # generate bounding box around object
            x, y, w, h = cv.boundingRect(approx)

            # categorize objects
            if objCor == 3:
                objectType = 'tri'
            elif objCor == 4:
                aspRatio = w / float(h)
                if 0.95 < aspRatio < 1.05:
                    objectType = 'square'
                else:
                    objectType = 'rect'
            else:
                objectType = 'circle'

            cv.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv.putText(imgContour, objectType, (x + (w // 2) - 12 - len(objectType), y + (h // 2)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


getContours(canny)
imgStack = stackImages(0.6, ([img, imgBlur], [canny, imgContour]))
cv.imshow('Stack', imgStack)

cv.waitKey(0)
