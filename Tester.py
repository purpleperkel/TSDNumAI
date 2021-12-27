from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import cv2

Path = "sample_images"

# load and prepare the image
def load_image(filename):
	# load the image
    img = load_img(filename, grayscale=True, target_size=(32, 32))
    #img = cv2.resize(img, (32, 32))
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 1 channel
    img = img.reshape(1, 32, 32, 1)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img
 
# load an image and predict the class
def run_example():

	model = load_model('ConvNet.h5')
	for item in os.listdir(Path):
		if item == '.DS_Store':
			continue
		filePath = Path+"/"+item
		if os.path.isfile(filePath):
			# load the image
			img = load_image(filePath)
			# load model
			# predict the class
			predict_value = model.predict(img)
			digit = argmax(predict_value)
			answer = str(digit)
			if (digit > 9):
				answer = str(chr(digit-10+65))

			print(filePath + ": " + answer)
		else:
			print("Error path: " + filePath)
 
# entry point, run the example
run_example()