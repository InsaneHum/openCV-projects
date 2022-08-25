import cv2 as cv

img = cv.imread('img/people2.JPG')

resized = cv.resize(img, (1500, 1000), interpolation=cv.INTER_AREA)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

# create a haar cascade variable
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# detects face using variables and return the rectangular coordinates of the face as a list to faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
# minNeighbors = the number of neighbors a rectangle should have to be called a face

print(f'Found {len(faces_rect)} faces.')

for (x, y, w, h) in faces_rect:
    # draw a rectangle over the found face based on the coordinates
    cv.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', resized)

cv.waitKey(0)
