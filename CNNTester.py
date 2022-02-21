from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf

model = load_model('CNN.h5')
classes = np.loadtxt("classNames.csv", dtype=str)
img = cv2.imread("CNN_test.jpg")

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
i = np.argmax(predictions)
prob = predictions[0][i]
label = classes[i]
print(label + " - " + str(prob*10))