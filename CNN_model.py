import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,Activation,Flatten,MaxPooling2D
from keras.callbacks import TensorBoard


NAME = "Cats-vs-dogs-CNN_with_dense"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


pickle_in_x = open("X.pickle","rb")
pickle_in_y = open("y.pickle","rb")
X = pickle.load(pickle_in_x)
y = pickle.load(pickle_in_y)

X = X/255.0


model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #converts 3d feature map to 1d

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X,y,batch_size=32,epochs=10,validation_split=0.1, callbacks=[tensorboard])


'''
HOW TO USE TensorBoard?
open terminal in the root folder "Cats-vs-dogs-CNN"
type the below command and enter:

tensorboard --logdir='logs/'

copy and paste the link in the browser
'''