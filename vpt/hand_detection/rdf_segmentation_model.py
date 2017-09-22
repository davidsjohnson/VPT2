import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import cv2

from vpt.common import *
import vpt.hand_detection.depth_context_features as dcf
import vpt.settings as s


class RDFSegmentationModel():

    def __init__(self, M, radius, n_samples=500, n_estimators=10, max_depth=20):

        self._M = M
        self._radius = radius
        self._n_samples = n_samples

        self._offsets = dcf.generate_feature_offsets(self._M, self._radius)

        self._clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


    def train(self, folders):

        mask_paths, orig_paths = self.get_fpaths(folders)

        n_features = len(self._offsets)

        ########
        # Generate the Dataset for training
        ########
        # X_lh = np.array([[]]).reshape((0, n_features))
        # y_lh = np.array([])
        #
        # X_rh = np.array([[]]).reshape((0, n_features))
        # y_rh = np.array([])
        #
        # X_bg = np.array([[]]).reshape((0, n_features))
        # y_bg = np.array([])

        X_lh = []
        y_lh = []

        X_rh = []
        y_rh = []

        X_bg = []
        y_bg = []

        for i, (mask_path, orig_path) in enumerate(zip(mask_paths, orig_paths)):

            mask = np.load(mask_path)
            orig = np.load(orig_path)

            print ("Extracting Features from Image:", i)
            start_time = time.time()
            lh_results = self.extract_features(mask[:, :, s.LH], orig, s.LH_LBL, self._offsets, self._n_samples)
            X_lh.append(lh_results[0])
            y_lh.append(lh_results[1])

            rh_results = self.extract_features(mask[:, :, s.RH], orig, s.RH_LBL, self._offsets, self._n_samples)
            X_rh.append(rh_results[0])
            y_rh.append(rh_results[1])

            bg_mask = np.logical_not(mask[:, :, s.LH] + mask[:, :, s.RH])         # combine left and right to find background
            bg_results = self.extract_features(bg_mask, orig, s.BG_LBL, self._offsets, self._n_samples)
            X_bg.append(bg_results[0])
            y_bg.append(bg_results[1])

            print ("LH", np.array(lh_results[0]).shape)
            print ("RH", np.array(rh_results[0]).shape)

            if rh_results[0].shape[0] != 500 or lh_results[0].shape[0] != 500 or rh_results[0].shape[1] != 48 or lh_results[0].shape[1] != 48:
                print ("ERROR::::Invalid array size in file:", mask_path, orig_path)

            end_time = time.time()
            total_time = end_time-start_time
            print ("\tDone: Took %f seconds" % total_time)

        print ("XLH:", np.array(X_lh).shape)
        print ("XRH:", np.array(X_rh).shape)
        print ("XBG:", np.array(X_bg).shape)

        X = np.concatenate((X_lh, X_rh, X_bg))
        y = np.concatenate((y_lh, y_rh, y_bg))

        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        y = y.reshape(y.shape[0]*y.shape[1])

        print ("X:", X.shape)
        print ("y:", y.shape)

        # Data set generated
        ##########

        ##########
        # Train
        ##########
        print ("Training Hand Segmentation Model")
        self._clf.fit(X, y)


    def generate_mask(self, depth_map):

        mask_shape = (depth_map.shape[0], depth_map.shape[1], 3)
        mask = np.zeros(mask_shape, dtype="uint8")

        all_features1 = dcf.calc_features(depth_map, self._offsets)

        p = self._clf.predict(np.array(all_features1).squeeze())
        p = p.reshape(depth_map.shape)

        mask[:, :, s.LH][p == s.LH_LBL] = 255
        mask[:, :, s.RH][p == s.RH_LBL] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        mask[:, :, s.LH] = cv2.morphologyEx(mask[:, :, s.LH], cv2.MORPH_OPEN, kernel)
        mask[:, :, s.RH] = cv2.morphologyEx(mask[:, :, s.RH], cv2.MORPH_OPEN, kernel)

        return mask


    def extract_features(self, mask, orig, label, offsets, n_samples=500):

        start_time = time.time()
        #### Generate Samples from Image ####
        # Generate locations of pixels for each hand and background

        pixels = np.where(mask > 0)
        pixels = np.hstack((np.expand_dims(pixels[1], 1), np.expand_dims(pixels[0], 1)))  # combine results to x, y pairs

        # Generate N samples from data set
        sample_idxs = np.arange(pixels.shape[0])
        np.random.shuffle(sample_idxs)
        sample_idxs = sample_idxs[:n_samples]

        sample_pixels = pixels[sample_idxs]
        sample_mask = np.zeros_like(orig, dtype=bool)

        sample_mask[sample_pixels[:, 1:], sample_pixels[:, :1]] = True

        X = dcf.calc_features(orig, offsets, sample_mask)
        y = np.ones((X.shape[0],))*label

        end_time = time.time()
        total_time = end_time - start_time
        print ("\tTotal Time for Samples: %f seconds: " % total_time)

        return X, y


    def get_fpaths(self, folders):

        origs = []
        masks = []

        for folder in folders:

            for root, sub_dirs, files in os.walk(folder):
                for filename in files:
                    if "mask" in filename:
                        masks.append(os.path.join(root, filename))
                    elif "orig" in filename:
                        origs.append(os.path.join(root, filename))
                    else:
                        print ("Invalid File Type")

            assert len(masks) == len(origs), "Assertion Error: Masks and Originals are not the same lengths"

        return np.array(masks), np.array(origs)
