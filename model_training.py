import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pickle
import matplotlib as plt

from tensorflow import keras

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


INIT_LR = 1e-4
EPOCHS = 10
BS = 32

trainX = pickle.load(open("trainX.pickle", "rb"))
trainY = pickle.load(open("trainY.pickle", "rb"))
testX = pickle.load(open("testX.pickle", "rb"))
testY = pickle.load(open("testY.pickle", "rb"))



baseModel = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape=(224,224,3)))
baseModel.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(64, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(32, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation = "sigmoid")(headModel)

model = keras.Model(inputs=baseModel.input, outputs=headModel)
model.summary()

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

H = model.fit(trainX,
              trainY,
              epochs=EPOCHS,
              validation_data=(testX, testY),
              batch_size=BS, verbose=2)

model.save('my_model')

# evaulate
model.evaluate( testX,
                testY,
                batch_size=BS,
                verbose=2)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/acc")
plt.legend(loc = "lower left")
plt.savefig("plot.png")