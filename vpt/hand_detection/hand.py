import numpy as np
import cv2

class Hand:

    def __init__(self, mask, depthmap, hand_box, fpath=None, label=None):

        self._mask          = mask
        self._depthmap      = depthmap
        self._hand_box      = hand_box
        self._fpath         = fpath

        self._label         = label


    def get_hand_img(self):

        x1 = self._hand_box[0]
        y1 = self._hand_box[1]
        x2 = x1 + self._hand_box[2]
        y2 = y1 + self._hand_box[3]

        masked = self._depthmap.copy()
        masked[self._mask == 0] = 0

        masked = cv2.GaussianBlur(masked, (3, 3), sigmaX=0, sigmaY=0)

        return masked[y1:y2, x1:x2]


    def get_fpath(self):
        return self._fpath


    def get_mask(self):
        return self._mask


    def get_original(self):
        return self._depthmap


    def dmap(self, dmap=None):
        if dmap is None:
            return self._depthmap
        else:
            self._depthmap = dmap


    def mask(self, mask=None):
        if mask is None:
            return self._mask
        else:
            self._mask = mask


    def label(self, label = None):
        if label is None:
            return int(self._label)
        else:
            self._label = label

    def hand_box(self, box=None):
        if box is None:
            return self._hand_box
        else:
            self._hand_box = box