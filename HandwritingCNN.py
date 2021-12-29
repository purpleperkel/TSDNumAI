# baseline cnn model for mnist
from numpy import mean, mod
from numpy import std
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import cv2
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	data = list()
	labels = list()
	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))
		# update the list of data and labels
		data.append(image)
		labels.append(label)
	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")
	# return a 2-tuple of the A-Z data and labels
	return (data, labels)

# load train and test dataset
def load_mnist_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	data = np.vstack([trainX, testX])
	labels = np.hstack([trainY, testY])
	#return trainX, trainY, testX, testY
	return (data, labels)
 
# scale pixels
def prep_pixels(data):
	# convert from integers to floats
	train_norm = [cv2.resize(image, (32, 32)) for image in data]
	#train_norm = train_norm.astype('float32')
	train_norm = np.array(train_norm, dtype="float32")
	# normalize to range 0-1
	train_norm = np.expand_dims(train_norm, axis=-1)
	train_norm = train_norm / 255.0
	# return normalized images
	return train_norm
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 1)))
	#model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	#model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	#model.add(BatchNormalization())
	model.add(Dense(36, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	print(model.summary())
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# evaluate a model
def evaluate_model(dataX, dataY):
	scores, histories = list(), list()
	model = define_model()
	(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.20, random_state=42)
	history = model.fit(
	aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // 32,
	epochs=3,
	verbose=0)
	model.save("ConvNet.h5")
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# stores scores
	scores.append(acc)
	histories.append(history)
	return scores, histories
 
# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()
 
# run the test harness for evaluating a model
def run_test_harness():
	dataPath = "HandwrittenData.csv"
	# load dataset
	(azData, azLabels) = load_az_dataset(dataPath)
	(digitsData, digitsLabels) = load_mnist_dataset()
	azLabels += 10
	
	data = np.vstack([azData, digitsData])
	labels = np.hstack([azLabels, digitsLabels])
	data = prep_pixels(data)

	# one hot encode target values
	labels = to_categorical(labels)
	# evaluate model
	scores, histories = evaluate_model(data, labels)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()