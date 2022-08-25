import os
import cv2 as cv
import numpy as np

# create a haar cascade variable
haar_cascade = cv.CascadeClassifier('haar_face.xml')

dir = r'C:\Users\Hum\PycharmProjects\openCV\train'

p = []
for i in os.listdir(dir):
    p.append(i)


features = []  # image arrays of faces
labels = []  # label for the images


def create_train():
    for person in p:
        path = os.path.join(dir, person)
        label = p.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print('Training done')

# convert features and labels to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# create face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# save the trained algorithm
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

