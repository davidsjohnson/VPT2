import numpy as np

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

        return masked[y1:y2, x1:x2]


    def label(self, label = None):
        if label == None:
            return int(self._label)
        else:
            self._label = label


    def get_fpath(self):
        return self._fpath