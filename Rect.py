import numpy as np
import cv2
from PIL import Image
import math

def distance(pt1, pt2):
    (x1, y1) = pt1
    (x2, y2) = pt2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# read image
img = cv2.imread("rectangle.png")

# convert img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = 255-gray

# do adaptive threshold on gray image
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
ret1,thresh = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
thresh = 255-thresh

# apply morphology
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

# separate horizontal and vertical lines to filter out spots outside the rectangle
kernel = np.ones((7,3), np.uint8)
vert = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3,7), np.uint8)
horiz = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# combine
rect = cv2.add(horiz,vert)
# thin
kernel = np.ones((3,3), np.uint8)

# fill in holes
rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

# preform edge detection
edged = cv2.GaussianBlur(rect, (5, 5), 0)
edged = cv2.Canny(edged, 50, 250)

# smooth ou tedges and try to close any open contours
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dilated = cv2.dilate(edged, kernel)
cv2.imshow('edges', dilated)

# get contours
contours = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# draw contours
imcopy = img.copy()
cv2.drawContours(imcopy, contours, -1, (0, 0, 255), 3)
cv2.imshow('contours', imcopy)

# get largest contour
area_thresh = 0
big_contour = None
i = 0
for c in contours:
    # collect area ratio of image and contour
    # as well as contour aspect ratio
    (ix, iy) = thresh.shape
    imgArea = ix*iy
    area = cv2.contourArea(c)
    imgArea = area/imgArea
    rot_rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    distances = list()
    distances.append(distance(tuple(box[0]), tuple(box[1])))
    distances.append(distance(tuple(box[0]), tuple(box[2])))
    distances.append(distance(tuple(box[0]), tuple(box[3])))
    l = min(distances)
    distances.remove(l)
    w = min(distances)
    ar = l/w

    # find largest contour that fits the aspect ratio and 
    # area ratio of a pan
    i += 1
    contours1 = c.reshape(-1,2)
    if (area > area_thresh and ar < .52 and ar > .4 and imgArea > .007 and imgArea < .05):
        area_thresh = area

        big_contour = c

# get rotated rectangle from contour
if (big_contour is not None):
    rot_rect = cv2.minAreaRect(big_contour)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    print(box)

    # draw rotated rectangle on copy of img
    rot_bbox = img.copy()
    cv2.drawContours(rot_bbox,[box],0,(0,0,255),2)

    # write img with red rotated bounding box to disk
    cv2.imwrite("output_images/rectangle_thresh.png", thresh)
    cv2.imwrite("output_images/rectangle_outline.png", rect)
    cv2.imwrite("output_images/rectangle_bounds.png", rot_bbox)

    # display it
    cv2.imshow("IMAGE", img)
    cv2.imshow("THRESHOLD", thresh)
    cv2.imshow("MORPH", morph)
    cv2.imshow("VERT", vert)
    cv2.imshow("HORIZ", horiz)
    cv2.imshow("RECT", rect)
    cv2.imshow("BBOX", rot_bbox)
cv2.waitKey(0)