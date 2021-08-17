import time

import cv2
import numpy as np


print("Face Identification Program is running...")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    print(img.shape)
    #turn it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Detect the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #show the image
    cv2.imshow('img', img)
    
    #break out of loop on keystroke
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
#close the window that shows the image
cv2.destroyAllWindows()