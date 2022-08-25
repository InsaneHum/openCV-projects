import cv2 as cv

capture = cv.VideoCapture(0)

# create a haar cascade variable
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detects face using variables and return the rectangular coordinates of the face as a list to faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    # minNeighbors = the number of neighbors a rectangle should have to be called a face

    for (x, y, w, h) in faces_rect:
        # draw a rectangle over the found face based on the coordinates
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):  # if 'd' key is pressed, terminate loop
        break

capture.release()
cv.destroyAllWindows()
