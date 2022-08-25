import cv2 as cv
import numpy as np

frameWidth = 640
frameHeight = 320
capture = cv.VideoCapture(0)  # 0 for webcam
capture.set(3, frameWidth)
capture.set(4, frameHeight)
capture.set(10, 150)  # brightness

# values for blue glue
myColors = [[61, 175, 151, 125, 255, 255],
            [29, 68, 61, 45, 150, 255],
            [162, 92, 126, 180, 188, 255],
            [62, 62, 148, 86, 188, 255]]

myColorValues = [[255, 0, 0],  # in BGR
                 [0, 255, 255],
                 [0, 0, 255],
                 [0, 255, 0]]

myPoints = []  # [x, y, colorID]


def findColor(image, colors, colorValues):
    imgHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in colors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)

        if x != 0 and y != 0:
            cv.circle(imgResult, (x, y), 10, colorValues[count], cv.FILLED)
            newPoints.append([x, y, count])
        count += 1
        # cv.imshow(str(color[0]), mask)

    return newPoints


def getContours(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:
            # cv.drawContours(imgResult, cnt, -1, (255, 0, 0), 6)
            # canvas, contour, how many contours (-1 for all), color, thickness

            # calculate arclength to find corners (contour, is shape closed)
            peri = cv.arcLength(cnt, True)

            # approx the corner points of the shape
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            # generate bounding box around object
            x, y, w, h = cv.boundingRect(approx)

    return x + w//2, y


def drawOnCanvas(points, colorValues):
    for point in points:
        cv.circle(imgResult, (point[0], point[1]), 10, colorValues[point[2]], cv.FILLED)


# # use trackbars to find target HSV values
# cv.namedWindow('TrackBars')
# cv.resizeWindow('TrackBars', 640, 250)
# cv.createTrackbar('Hue Min', 'TrackBars', 61, 180, lambda: 0)
# cv.createTrackbar('Hue Max', 'TrackBars', 125, 180, lambda: 0)
# cv.createTrackbar('Sat Min', 'TrackBars', 175, 255, lambda: 0)
# cv.createTrackbar('Sat Max', 'TrackBars', 255, 255, lambda: 0)
# cv.createTrackbar('Val Min', 'TrackBars', 151, 255, lambda: 0)
# cv.createTrackbar('Val Max', 'TrackBars', 255, 255, lambda: 0)d


# use a while loop to read the video frame by frame
while True:
    isTrue, frame = capture.read()
    # flip the screen
    frame = cv.flip(frame, 1)
    imgResult = frame.copy()
    # cv.imshow('Video', img)

    # # convert image to HSV to detect color
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #
    # h_min = cv.getTrackbarPos('Hue Min', 'TrackBars')
    # h_max = cv.getTrackbarPos('Hue Max', 'TrackBars')
    # s_min = cv.getTrackbarPos('Sat Min', 'TrackBars')
    # s_max = cv.getTrackbarPos('Sat Max', 'TrackBars')
    # v_min = cv.getTrackbarPos('Val Min', 'TrackBars')
    # v_max = cv.getTrackbarPos('Val Max', 'TrackBars')
    #
    # # print(h_min, h_max, s_min, s_max, v_min, v_max)
    #
    # # filter the images based on the HSV values
    # lower = np.array([h_min, s_min, v_min])
    # upper = np.array([h_max, s_max, v_max])
    # mask = cv.inRange(hsv, lower, upper)
    #
    # imgResult = cv.bitwise_and(img, img, mask=mask)
    #
    # stacked = np.hstack((np.vstack((img, imgResult)), np.vstack((hsv, np.dstack((mask, mask, mask))))))
    #
    # cv.imshow('Stacked', stacked)

    foundPoints = findColor(frame, myColors, myColorValues)
    if len(foundPoints) != 0:
        for newP in foundPoints:
            myPoints.append(newP)

    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)

    cv.imshow('Results', imgResult)

    if cv.waitKey(1) & 0xFF == ord('q'):  # if 'q' key is pressed, clear canvas
        myPoints = []

    if cv.waitKey(20) & 0xFF == ord('d'):  # if 'd' key is pressed, terminate loop
        break

capture.release()
cv.destroyAllWindows()
