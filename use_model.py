import cv2
import tensorflow as tf
from tensorflow import keras

Categories = ["Dog", "Cat"]

def prepare(filepath):
    img_size = 90
    try:
        path=filepath
        img_array=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        new_array=cv2.resize(img_array, (img_size,img_size))
        return new_array.reshape(-1,img_size,img_size,1)
    except Exception as e:
        print(str(e))


model = keras.models.load_model("64*3-CNN.model")

prediction = model.predict([prepare("/Users/priyanshumadan/PycharmProjects/cat-vs-dogs-CNN/dog.jpeg")])
print(Categories[int(prediction[0][0])])
