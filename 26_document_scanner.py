import cv2 as cv
import numpy as np

frameWidth = 715
frameHeight = 500
capture = cv.VideoCapture(0)  # 0 for webcam
capture.set(3, frameWidth)
capture.set(4, frameHeight)
capture.set(10, 150)  # brightness


def preProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 175, 175)
    # dilate and erode edges for better results
    kernel = np.ones((5, 5))
    imgDial = cv.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv.erode(imgDial, kernel, iterations=1)

    return imgThres
    

def getContours(image):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 5000:
            # calculate arclength to find corners (contour, is shape closed)
            peri = cv.arcLength(cnt, True)
            # approx the corner points of the shape
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    # canvas, contour, how many contours (-1 for all), color, thickness

    return biggest


# function to orientate points
def reorder(contourPts):
    contourPts = contourPts.reshape((4, 2))
    newPts = np.zeros((4, 1, 2), np.int32)

    add = contourPts.sum(1)
    newPts[0] = contourPts[np.argmin(add)]
    newPts[3] = contourPts[np.argmax(add)]

    diff = np.diff(contourPts, axis=1)
    newPts[1] = contourPts[np.argmin(diff)]
    newPts[2] = contourPts[np.argmax(diff)]

    return newPts


def getWarp(image, contourPts):
    contourPts = reorder(contourPts)
    pts1 = np.float32(contourPts)
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(image, matrix, (frameWidth, frameHeight))

    imgCropped = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]  # crop 10 pixels off each direction
    imgCropped = cv.resize(imgCropped, (frameWidth, frameHeight))  # resize the image back to its original size

    return imgCropped
    

while True:
    isTrue, frame = capture.read()
    # flip the screen
    frameFlipped = cv.flip(frame, 1)

    imgContour = frame.copy()
    imgProcessed = preProcessing(frame)
    pts = getContours(imgProcessed)
    if pts.size != 0:
        imgWarped = getWarp(frame, pts)
        cv.imshow('Warped', imgWarped)

    stacked = np.hstack((imgContour, np.dstack((imgProcessed, imgProcessed, imgProcessed))))
    cv.imshow('Stacked', stacked)

    if cv.waitKey(20) & 0xFF == ord('d'):  # if 'd' key is pressed, terminate loop
        break

capture.release()
cv.destroyAllWindows()
