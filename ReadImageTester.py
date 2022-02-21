from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import Rect

model = load_model('ConvNet.h5')
panMode = False

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
    if (len(cnts) == 0):
        return None
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
       key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def modify_tuple(x, y, tup):
    ls = list(tup)
    ls[0] += x
    ls[1] += y
    return tuple(ls)

def get_edges(src, cdim, roipad, pan, tmp):
    edged = cv2.Canny(src, 50, 250)
    edgepic = np.array(edged)
    edgepic = Image.fromarray(edgepic.astype(np.uint8))
    edgepic.save('output_images/edgepic.jpeg', 'JPEG')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    dilated = cv2.dilate(edged, kernel)
    dpic = Image.fromarray(dilated.astype(np.uint8))
    dpic.save('output_images/dilated.jpeg', 'JPEG')
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    cnts = sort_contours(cnts, method="left-to-right")
    if (cnts is None):
        return None
    else:
        cnts = cnts[0]
    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []
    boxes = []

    (wx, wy, lx, ly) = cdim
    (pwx, pwy, plx, ply) = roipad
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        a = None
        b = None
        d = None
        if (not pan):
            (a,b) = src.shape
        else:
            (a,b,d) = src.shape
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w/a >= wx and w/a <= wy) and (h/b >= lx and h/b <= ly):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = None
            if (pan):
                roi = tmp[(y+pwx):y+h+pwy, (x+plx):x + w+ply]
            else:
                roi = src[(y+pwx):y+h+pwy, (x+plx):x + w+ply]
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

            if (not pan):
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
            if (not pan):
                thresh = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            if (pan):
                ckimg2 = Image.fromarray(thresh)
                ckimg2.show()
                # apply morphological methods to make sure number contours will be closed
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
                erosion = cv2.erode(thresh,kernel,iterations = 1)
                erosion = cv2.dilate(thresh,kernel,iterations = 1)
                erosion = cv2.erode(thresh,kernel,iterations = 1)
                morphpic = Image.fromarray(erosion.astype(np.uint8))
                morphpic.save('output_images/morph.jpeg', 'JPEG')
                thresh = cv2.GaussianBlur(erosion, (5, 5), 0)
                (chars, boxes1) = get_edges(thresh, (0, 1, 0, 1), (0, 0, 0, 0), False, None)
                boxes = list(map(lambda t: modify_tuple(x+pwx,y+plx,t), boxes1))
            else:
                # append each character found
                thresh = cv2.bitwise_not(thresh)
                ckimg2 = Image.fromarray(thresh)
                ckimg2.show()
                thresh = np.expand_dims(thresh, axis=-1)
                thresh = thresh.reshape(1, 32, 32, 1)
                thresh = thresh.astype('float32')
                thresh = thresh / 255.0
                chars.append(thresh)
                boxes.append((x,y,w,h))
    return (chars, boxes)

def get_debug_pred(debug_flag):
    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    image = cv2.imread("pan.jpg")

    #img = cv2.imread("pan.jpg")
    # resize image
    image = cv2.resize(image, (0,0), fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
    # put non-pan color border to ensure good pan contour
    image = cv2.copyMakeBorder(
        image, 
        10, 
        10, 
        10, 
        10, 
        cv2.BORDER_CONSTANT, 
        value=(0,255,0)
    )
    roi = Rect.get_pan(image, 0)
    if (roi is None):
        roi = Rect.get_pan(image, 1)
        if (roi is None):
            return None
    #cv2.waitKey(0)

    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = roi
    temp = gray
    if (panMode):
        # apply mask to get pan and reduce noise
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #lower 80 upper 120 for 4000s
        lower_gray = np.array([0, 0, 140], np.uint8)
        upper_gray = np.array([0, 0, 200], np.uint8)
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        img_res = cv2.bitwise_and(image, image, mask = mask)
        gray = img_res

    cv2.imwrite('output_images/gray.png',gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite('output_images/blurred.png',blurred)
    #if (panMode):
        #(chars, boxes) = get_edges(blurred, (.25, .45, .25, .35), (11, -4, 4, -4), True, temp)
    #    (chars, boxes) = get_edges(blurred, (.3, .55, .25, .35), (11, -8, 10, -4), True, temp)
    #else:
    edges = get_edges(blurred, (.05, .8, .05, .9), (-2, 2, -2, 2), False, None)
    (chars, boxes) = edges
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

def get_pred(src, debug_flag):
    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    image = cv2.imread(src)

    #img = cv2.imread("pan.jpg")
    # resize image
    image = cv2.resize(image, (0,0), fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
    # put non-pan color border to ensure good pan contour
    image = cv2.copyMakeBorder(
        image, 
        10, 
        10, 
        10, 
        10, 
        cv2.BORDER_CONSTANT, 
        value=(0,255,0)
    )
    roi = Rect.get_pan(image, 0)
    if (roi is None):
        print("trying mode 1")
        roi = Rect.get_pan(image, 1)
        if (roi is None):
            return None
    #cv2.waitKey(0)

    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = roi
    temp = gray
    #cv2.imwrite('output_images/gray.png',gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #cv2.imwrite('output_images/blurred.png',blurred)
    #if (panMode):
        #(chars, boxes) = get_edges(blurred, (.25, .45, .25, .35), (11, -4, 4, -4), True, temp)
    #    (chars, boxes) = get_edges(blurred, (.3, .55, .25, .35), (11, -8, 10, -4), True, temp)
    #else:
    edges = get_edges(blurred, (0, .8, .1, .9), (-2, 2, -2, 2), False, None)

    if (edges is None):
        return "oh no :c"

    (chars, boxes) = edges
    # OCR the characters using our handwriting recognition model
    preds = list(map(model.predict, chars))
    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]
    return_string=""

    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[0][i]
        label = labelNames[i]
        return_string = return_string+label
        # draw the prediction on the image
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(image, label, (x - 10, y - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        # show the image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
    
    return return_string

get_debug_pred(True)