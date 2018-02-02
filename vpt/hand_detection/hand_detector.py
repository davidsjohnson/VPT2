import cv2

from vpt.hand_detection.sub_segmentation_model import *
from vpt.hand_detection.hand import Hand

from vpt.common import *
import vpt.settings as s

import time

class HandDetector():

    def __init__(self, hs_model):

        self._hs_model = hs_model


    def detect_hands(self, depthmap, fpath=None, labels=(None, None)):

        lh_label = s.ANNOTATIONS.index(labels[s.LH])
        rh_label = s.ANNOTATIONS.index(labels[s.RH])

        if isinstance(self._hs_model, list):

            if "error_0" in fpath:
                hs_idx = 0
            elif "error_1" in fpath:
                hs_idx = 1
            elif "error_2" in fpath:
                hs_idx = 2
            else:
                hs_idx = 10

            mask = self._hs_model[hs_idx].generate_mask(depthmap)

        else:
            mask = self._hs_model.generate_mask(depthmap)

        mask = cv2.GaussianBlur(mask, (5,5), sigmaX=0, sigmaY=0)

        # get bounding boxes of hands from mask
        lh_box = self.get_bounding_box(mask[:, :, s.LH].copy())  # added .copy() to remove error with findContours
        rh_box = self.get_bounding_box(mask[:, :, s.RH].copy())

        if lh_box == None or rh_box == None:
            raise ValueError("No Hands detected")

        # create Hand Objects to return
        lh = Hand(mask[:, :, s.LH], depthmap, lh_box, fpath, lh_label)
        rh = Hand(mask[:, :, s.RH], depthmap, rh_box, fpath, rh_label)

        return lh, rh


    def get_bounding_box(self, mask):

        if cv2.__version__ < '3.0':
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = 0
        max_rect = None

        for c in contours:

            # find bounding box and area
            poly = cv2.approxPolyDP(c, epsilon=3, closed=True)
            rect = cv2.boundingRect(poly)
            area = rect[2]*rect[3]

            if area > max_area:
                max_area = area
                max_rect = rect

        return max_rect