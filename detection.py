import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array

from keras.applications.mobilenet_v2 import preprocess_input

print("Face Identification Program is running...")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = keras.models.load_model('./my_model')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    #turn it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Detect the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, dsize = (224, 224))
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        resized_roi = roi.reshape(1,224,224,3)
        pred = model.predict(resized_roi)[0][0]
        print(str(pred))
        if pred < 0.9:
            print("wearing mask")
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            print("not wearing mask")
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #show the image
    cv2.imshow('img', img)
    
    #break out of loop on keystroke
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
#close the window that shows the image
cv2.destroyAllWindows()