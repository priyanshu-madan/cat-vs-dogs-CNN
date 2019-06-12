#LOADING YOUR OWN DATA

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

# for category in categories:
#     path = os.path.join(data_dir,category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array,cmap="gray")
#         plt.show()
#         break
#     break

img_size = 90 #set image size

# view the image with the scale set above
#NOT PART OF ACTUAL MODEL

# new_array = cv2.resize(img_array,(img_size,img_size))
# plt.imshow(new_array,cmap="gray")
# plt.show()


training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, img_size, img_size, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
