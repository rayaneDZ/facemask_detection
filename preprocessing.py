import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pickle

from tensorflow import keras

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.applications.mobilenet_v2 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATADIR = "./faces"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224)) #scaling the pixel intensities in the input image to the range [-1, 1]
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, stratify = labels,test_size = 0.20, random_state = 42) #also shuffles automatically

trainY = trainY.reshape(trainY.shape[0], 1)
testY = testY.reshape(testY.shape[0], 1)

pickle_out = open("./pickeled_data/trainX.pickle", "wb")
pickle.dump(trainX, pickle_out, protocol = 4)
pickle_out.close()

pickle_out = open("./pickeled_data/testX.pickle", "wb")
pickle.dump(testX, pickle_out, protocol = 4)
pickle_out.close()

pickle_out = open("./pickeled_data/trainY.pickle", "wb")
pickle.dump(trainY, pickle_out, protocol = 4)
pickle_out.close()

pickle_out = open("./pickeled_data/testY.pickle", "wb")
pickle.dump(testY, pickle_out, protocol = 4)
pickle_out.close()