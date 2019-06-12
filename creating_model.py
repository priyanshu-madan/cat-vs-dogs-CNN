import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import pickle
import tensorflow as tf
from tensorflow import keras

data_dir = "/Users/priyanshumadan/PycharmProjects/cat-vs-dogs-CNN/PetImages"

categories = ["Dog","Cat"]

#to view the original images in the directory in grayscale
#NOT PART OF ACTUAL MODEL
for category in categories:
    path = os.path.join(data_dir,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break

img_size = 90 #set image size

# view the image with the scale set above
#NOT PART OF ACTUAL MODEL
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap="gray")
plt.show()

