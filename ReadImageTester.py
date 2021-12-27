from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import argparse
import cv2

model = load_model('ConvNet.h5')

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
image = cv2.imread("sample.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png',gray)
#gray = img_res

#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv2.Canny(blurred, 60, 250)
edgepic = np.array(edged)
edgepic = Image.fromarray(edgepic.astype(np.uint8))
edgepic.save('edgepic.jpeg', 'JPEG')
#np.reshape(edgepic, (192,192))
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) == 2:
    cnts = cnts[0]

# if the length of the contours tuple is '3' then we are using
# either OpenCV v3, v4-pre, or v4-alpha
elif len(cnts) == 3:
    cnts = cnts[1]

cnts = sort_contours(cnts, method="left-to-right")[0]
# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []
boxes = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    (a,b,c) = image.shape
	# filter out bounding boxes, ensuring they are neither too small
	# nor too large
    if (w/a >= 0 and w/a <= .5) and (h/b >= .1 and h/b <= .8):
        # extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[(y-2):y+h+2, (x-2):x + w+2]
        #roi = gray[(y):y + h, (x):x + w]
        thresh = cv2.threshold(roi, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if (thresh is None or (np.all(thresh==0))):
            continue 
        (tH, tW) = thresh.shape
        dim = None
        (hi, wi) = thresh.shape[:2]
        if tW < tH:
            r = 32 / float(hi)
            z = int(wi * r)
            dim = ([z, 1][(z == 0)], 32)
        # otherwise, resize along the height
        else:
            r = 32 / float(wi)
            z = int(hi * r)
            dim = (32, [z, 1][(z == 0)])

        thresh = cv2.resize(thresh, dim)

        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        # pad the image and force 32x32 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255))
        thresh = cv2.resize(padded, (32, 32))
        # prepare the padded image for classification via our
        # handwriting OCR model
        #padded = padded.astype("float32") / 255.0
        #padded = np.expand_dims(padded, axis=-1)
        # update our list of characters that will be OCR'd
        #thresh = np.reshape(thresh, -1)
        #thresh = np.expand_dims(thresh, axis=-1)
        
        #thresh = np.reshape(thresh, (1, 32, 32, 1))
        #padded = np.reshape(padded, (1, 32, 32, 1))
        #thresh = Image.fromarray(thresh)
        #thresh.show()
        
        #thresh = np.asarray(thresh)
        #tmp = np.zeros(32*32).reshape(1,32,32,1)
        thresh = cv2.bitwise_not(thresh)
        ckimg2 = Image.fromarray(thresh)
        ckimg2.show()
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1, 32, 32, 1)
        thresh = thresh.astype('float32')
        thresh = thresh / 255.0
        chars.append(thresh)
        boxes.append((x,y,w,h))

        #thresh = np.expand_dims(thresh, axis=-1)
        #thresh = thresh.reshape(1, 32, 32, 1)
        #thresh = thresh.astype('float32')
        #thresh = thresh / 255.0
        #thresh = tmp + thresh
        #thresh = np.array(thresh, dtype="float32")
        #cv2.imshow("Image", thresh)
        #ckimg = Image.fromarray(thresh)
        #ckimg.save("tmp.png")
        #thresh = np.expand_dims(thresh, axis=-1)

        #img = load_img(thresh)
        #chars.append(thresh)
        #boxes.append((x,y,w,h))
        #end if

# extract the bounding box locations and padded characters
#boxes = [b[1] for b in chars]
#chars = np.array([c[0] for c in chars], dtype="float32")
# OCR the characters using our handwriting recognition model
preds = list(map(model.predict, chars))
# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[0][i]
    label = labelNames[i]
    # draw the prediction on the image
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)