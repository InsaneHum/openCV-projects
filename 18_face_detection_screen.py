import cv2 as cv
import numpy as np
from PIL import ImageGrab

capture = cv.VideoCapture(0)

# create area to capture screen
bounding_box = (0, 100, 1800, 1200)

# create a haar cascade variable
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    img = ImageGrab.grab(bounding_box)
    frame = np.array(img)
    # isTrue, frame = capture.read()

    rgb = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detects face using variables and return the rectangular coordinates of the face as a list to faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    # minNeighbors = the number of neighbors a rectangle should have to be called a face

    for (x, y, w, h) in faces_rect:
        # draw a rectangle over the found face based on the coordinates
        cv.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.putText(rgb, f'Found {len(faces_rect)} faces.', (50, 50), cv.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), thickness=3)

    cv.imshow('Detected Faces', rgb)

    if cv.waitKey(20) & 0xFF == ord('d'):  # if 'd' key is pressed, terminate loop
        break

capture.release()
cv.destroyAllWindows()
