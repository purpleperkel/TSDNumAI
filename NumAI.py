from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import *
from skimage.transform import resize
import os

#dataset_url = "TSDImageMerge/"
#data_dir = os.listdir( dataset_url )
data_dir = "TSDImageMerge/"


batch_size = 32
img_height = 192
img_width = 192

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = stripped = list(map(str.strip, train_ds.class_names))
#train_ds = array(train_ds).reshape(1, img_height, img_width, 3)
print(class_names)


model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2,2), activation='relu'),
    input_shape=(None, img_width, img_height,1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

#X, y = list(), list()
model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=0)