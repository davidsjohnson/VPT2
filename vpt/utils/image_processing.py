from vpt.common import *
import cv2

def color_hands(img):

    boxes = parse_hands(img)

    x1 = boxes[0][0]
    y1 = boxes[0][1]
    x2 = boxes[0][2]
    y2 = boxes[0][3]

    img[y1:y2, x1:x2, 0] = 255

    x1 = boxes[1][0]
    y1 = boxes[1][1]
    x2 = boxes[1][2]
    y2 = boxes[1][3]

    img[y1:y2, x1:x2, 1] = 255

    return img


def parse_hands(img):
    ''' Finds each hand from the given thresholded image.  Returns one image
        containing the left hand and one image containing the right hand
    '''

    rects, poly = bounding_rects(img)	# find the bounding rectangles of each hand (and polygons)

    # smallest X coordinate is the right hand
    if rects[0][0] < rects[1][0]:
        right_hand = rects[0]
        left_hand = rects[1]
    else:
        right_hand = rects[1]
        left_hand = rects[0]

    # get full set of left hand coordinates
    left_x1 = left_hand[0]
    left_y1 = left_hand[1]
    left_x2 = left_x1 + left_hand[2]
    left_y2 = left_y1 + left_hand[3]

    # get full set of right hand coordinates
    right_x1 = right_hand[0]
    right_y1 = right_hand[1]
    right_x2 = right_x1 + right_hand[2]
    right_y2 = right_y1 + right_hand[3]

    # return the left hand image, return the right hand image
    boxes = [[],[]]
    boxes[0] = (left_x1, left_y1, left_x2, left_y2)
    boxes[1] = (right_x1, right_y1, right_x2, right_y2)
    return boxes


def bounding_rects(img):
    ''' Finds the bounding rectangles of the hands.  Returns the bounding rectangles for
        each hand and the polygon contours
    '''

    # edge detection parameters
    canny_thresh1 = 500
    canny_thresh2 = 600
    aperture_size = 5

    img = img.copy()
    img[:5, :] = 0

    # perform edge detection, used to find the image contours
    edges = cv2.Canny(img, canny_thresh1, canny_thresh2, apertureSize=aperture_size)
    # _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)	# OPENCV 3.1
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # OPENCV 2.4

    contour_polys = []
    rects = [[0, 0, 0, 0], [0, 0, 0, 0]]  # only storing rectangles for left and right hand

    # assuming two largest rectangles (by area) are the hands
    max2_area = 0
    max1_area = 0

    for i in range(len(contours)):  # for each image contour

        contour_polys.append(cv2.approxPolyDP(contours[i], 3, True))  # find the polygon (UPDATE: Research Parameters)
        rect = cv2.boundingRect(contour_polys[i])  # find the bounding rect
        area = rect[2] * rect[3]

        if area > max2_area:  # check if rect area greater than second largest area
            if area > max1_area:  # if it is, is it greater than first largest area?

                max2_area = max1_area  # if so move largest to second largest
                rects[1] = rects[0]

                max1_area = area  # update largest to new rect
                rects[0] = rect

            else:
                max2_area = area  # not bigger than first so update second with new rect
                rects[1] = rect

    return rects, contour_polys


def normalize(arr):

    max_val = 1300
    return arr.astype(float) * 1.0/max_val


def normalize2(arr):

    arr = arr.astype(float)

    min_val = arr[arr != 0].min()  # find min
    max_val = arr[arr != 0].max()  # find max

    arr[arr != 0] = (arr[arr != 0] - min_val) / (max_val - min_val)  # normalize all non zero values
    arr[arr!=0] = 1.0-arr[arr!=0]

    return arr
