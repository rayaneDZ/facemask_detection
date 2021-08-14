import cv2
import pickle
import sys
import matplotlib.pyplot as plt

trainX = pickle.load(open("./pickeled_data/trainX.pickle", "rb"))
trainY = pickle.load(open("./pickeled_data/trainY.pickle", "rb"))

for i in range(10, 15):
    print(trainY[i])
    plt.imshow(trainX[i])
    plt.show()
