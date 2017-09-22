from vpt.common import *
import cv2

import vpt.utils.image_processing as ip
import matplotlib.pyplot as plt
import vpt.settings as s


class SubSegmentationModel():

    def __init__(self, model_name):

        self.model_name = model_name
        self.initialized = False
        self.kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))


    def initialize(self, background_folder, ext="bin", history=100, varThreshold=.1, shadowDetection=False):

        self.fgbg = cv2.BackgroundSubtractorMOG2(history, varThreshold, shadowDetection)

        for root, dirs, files in os.walk(background_folder):
            for fname in files:
                if ext in fname:
                    try:
                        fpath = os.path.join(root, fname)
                        data = load_depthmap(fpath)
                        self.fgbg.apply(data)
                    except Exception as e:
                        print e

        self.initialized = True


    def generate_mask(self, depth_map):

        mask_shape = (depth_map.shape[0], depth_map.shape[1], 3)
        mask = np.zeros(mask_shape, dtype="uint8")

        fgmask = self.fgbg.apply(depth_map)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)  # perform Morphological opening to remove noise

        img_bs = depth_map.copy()
        img_bs[fgmask == 0] = 0  # apply foreground mask

        fgmask = self.remove_legs(img_bs)

        boxes = ip.parse_hands(fgmask)

        x1, y1, x2, y2 = boxes[s.LH]
        mask[y1:y2, x1:x2, s.LH][fgmask[y1:y2, x1:x2] > 0 ] = 255

        x1, y1, x2, y2 = boxes[s.RH]
        mask[y1:y2, x1:x2, s.RH][fgmask[y1:y2, x1:x2] > 0 ] = 255

        return mask


    def remove_legs(self, img):
        ''' Need for Kinect data...but can we generalize?'''

        img_max = img.max()
        img[img == 0] = np.iinfo(img.dtype).max     # get max value allowed by data type and set 0 to this for better hist analysis

        bins = 50
        mask = np.zeros(img.shape, 'uint8')

        hist, edges = np.histogram(img, range=(0, img_max), bins=bins)

        peak1 = 1
        while peak1 + 1 < len(hist) and (hist[peak1] <= hist[peak1 + 1] or hist[peak1] < 1000):
            peak1 += 1

        minima = peak1 + 2
        while minima + 1 < len(hist) and hist[minima] - hist[minima + 1] > 20:
            minima += 1

        minima = min(minima, 50)
        thresh_val = edges[minima]

        mask[img <= thresh_val] = 255

        return mask