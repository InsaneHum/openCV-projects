import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Gareth Bale', 'Cristiano Ronaldo', 'Lionel Messi', 'Wu Jing']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('img/ronaldo_messi.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30)
print(faces_rect)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]

    # predict with face recognizer

    label, confidence = face_recognizer.predict(faces_roi)  # confidence: the lower the better
    print(f'{people[label]} with a confidence of {confidence}')
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.putText(img, str(people[label]), (x, y + h + 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), thickness=3)

cv.imshow('Detected Face', img)

cv.waitKey(0)

