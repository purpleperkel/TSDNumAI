import numpy as np
import cv2
from PIL import Image
import math

def get_mask(src, color):
    (lowerLimit, upperLimit) = color
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(lowerLimit), np.asarray(upperLimit))
    img_res = cv2.bitwise_and(src, src, mask = mask)
    return img_res

def distance(pt1, pt2):
    (x1, y1) = pt1
    (x2, y2) = pt2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_largest_contour(mask, color):
    # convert img to grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray = 255-gray

    # do adaptive threshold on gray image
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshVal = 170
    if (color == "indigo" or color == "purple"):
        threshVal = 200
    ret1,thresh = cv2.threshold(gray,threshVal,255,cv2.THRESH_BINARY)
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
    imcopy = mask.copy()
    cv2.drawContours(imcopy, contours, -1, (0, 0, 255), 3)
    cv2.imwrite('output_images/dcontours.png', imcopy)

    # get largest contour
    area_thresh = 0
    big_contour = None
    i = 0
    bci = 0
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
        #contours1 = c.reshape(-1,2)
        if (area > area_thresh and ar > .4 and ar < .55 and imgArea > 0.01 and imgArea < .3):
            area_thresh = area
            bci = i
            big_contour = c
        i += 1

    # get rotated rectangle from contour
    if (big_contour is not None):
        rot_rect = cv2.minAreaRect(big_contour)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        #print(box)

        # draw rotated rectangle on copy of img
        rot_bbox = mask.copy()
        cv2.drawContours(rot_bbox,[box],0,(0,0,255),2)

        # write img with red rotated bounding box to disk
        cv2.imwrite("output_images/rectangle_thresh.png", thresh)
        cv2.imwrite("output_images/rectangle_outline.png", rect)
        cv2.imwrite("output_images/rectangle_bounds.png", rot_bbox)

        # display it
        cv2.imshow("IMAGE", mask)
        cv2.imshow("THRESHOLD", thresh)
        cv2.imshow("MORPH", morph)
        cv2.imshow("VERT", vert)
        cv2.imshow("HORIZ", horiz)
        cv2.imshow("RECT", rect)
        cv2.imshow("BBOX", rot_bbox)
        return (contours, bci)

def subimage(image, center, theta, width, height):

   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image

def get_pan(im):
    color_thresholds = {
        "yellow": ((20, 100, 200), (50, 255, 255)),
        "pink": ((160, 0, 200), (180, 90, 255)),
        "blue": ((90, 100, 180), (110, 255, 255)),
        "purple": ((110, 100, 200), (130, 255, 255)),
        "gray": ((90, 70, 150), (110, 255, 170)),
        "green": ((70, 100, 120), (90, 255, 255)),
        "indigo": ((100, 100, 200), (120, 255, 255)),
        "red": ((0, 100, 200), (20, 230, 255)),
        "black": ((100, 0, 0), (120, 255, 40)),
        "white": ((80, 0, 250), (100, 255, 255))
    }

    i = 0
    bci = 0
    con = None
    mask=None
    color = None
    for c in color_thresholds:
        i += 1
        print(c)
        #print(con)
        #con = get_largest_contour(get_mask(im, color_thresholds[c]))
        mask = get_mask(im, color_thresholds[c])
        cv2.imwrite("output_images/mask.png", mask)
        con = get_largest_contour(mask, c)
        if (con is not None):
            con, bci = con
            color = c
            break

    if (con is not None):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #print(box)

        maskarray = np.zeros_like(gray) # Create mask where white is what we want, black otherwise
        cv2.drawContours(maskarray, con, bci, 255, -1) # Draw filled contour in mask
        cv2.imshow("maskawwarCont", maskarray)
        out = np.zeros_like(im) # Extract out the object and place into output image
        out[maskarray == 255] = im[maskarray == 255]

        rect = cv2.minAreaRect(con[bci])
        rot_bbox = out.copy()
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rot_bbox,[box],0,(0,0,255),1)
        pt, sz, ang = rect
        x,y = pt
        distances = list()
        distances.append(distance(tuple(box[0]), tuple(box[1])))
        distances.append(distance(tuple(box[0]), tuple(box[2])))
        distances.append(distance(tuple(box[0]), tuple(box[3])))
        h = min(distances)
        distances.remove(h)
        w = min(distances)
        (w,h) = sz
        #x = x - w/2
        #y = y - h/2
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        print("x,y: " + str(x) +","+ str(y))
        print("w,h: " + str(w) +","+ str(h))
        #roi = out[y:y+h, x:x+w]
        roi=subimage(out, (x,y), ang+270, h,w)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = 255-gray

        # do adaptive threshold on gray image
        #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
        threshVal = 170
        if (color == "indigo" or color == "purple"):
            threshVal = 200
        ret1,thresh = cv2.threshold(gray,threshVal,255,cv2.THRESH_BINARY)
        #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        im_floodfill = thresh.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        height, width = thresh.shape[:2]
        #mask = np.zeros((height+2, width+2), np.uint8)
        # Floodfill from point (0, 0)
        im_floodfill = cv2.copyMakeBorder(
            im_floodfill, 
            10, 
            10, 
            10, 
            10, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
        if (im_floodfill[0][0] == 255):
            cv2.floodFill(im_floodfill, None, (0,0), 0)
        if (im_floodfill[height-2][0] == 255):
            cv2.floodFill(im_floodfill, None, (width,0), 0)
        if (im_floodfill[0][width-2] == 255):
            cv2.floodFill(im_floodfill, None, (0,height), 0)
        if (im_floodfill[height-2][width-2] == 255):
            cv2.floodFill(im_floodfill, None, (width, height), 0)
        # Invert floodfilled image
        #im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the f`oreground.
        #im_out = thresh | im_floodfill_inv
        #thresh = 255-thresh
        cv2.imshow("ROI", roi)
        cv2.imshow("ROITHRESH", thresh)
        cv2.imshow("ROIINFILL", im_floodfill)
        cv2.imshow("maskk", mask)
        cv2.imshow("out", out)
        cv2.imshow("box", rot_bbox)
        return im_floodfill

""""

color_thresholds = {
    "yellow": ((20, 100, 200), (50, 255, 255)),
    "pink": ((160, 0, 200), (180, 90, 255)),
    "black": ((100, 0, 0), (120, 255, 40)),
    "purple": ((110, 100, 200), (130, 255, 255)),
    "white": ((80, 0, 250), (100, 255, 255)),
    "gray": ((90, 70, 150), (110, 255, 170)),
    "green": ((70, 100, 120), (90, 255, 255)),
    "indigo": ((100, 100, 200), (120, 255, 255)),
    "blue": ((90, 100, 180), (110, 255, 255)),
    "red": ((0, 100, 200), (20, 230, 255))
}
"""
# read image
img = cv2.imread("pan.jpg")
# resize image
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
#roi=get_mask(img, color_thresholds["indigo"])
#get_largest_contour(roi)
roi = get_pan(img)
cv2.imshow("rect_final", roi)
cv2.waitKey(0)
