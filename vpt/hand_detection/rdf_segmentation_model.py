import time
import sys
sys.path.append("./")
from sklearn.ensemble import RandomForestClassifier

import vpt.hand_detection.depth_context_features as dcf
from vpt.streams.file_stream import *

import vpt.common as c
import vpt.settings as s
from vpt.streams.mask_stream3 import MaskStream3


class RDFSegmentationModel():

    def __init__(self, M, radius, n_samples=500, n_estimators=10, max_depth=20):

        self._M = M
        self._radius = radius
        self._n_samples = n_samples

        self._offsets = dcf.generate_feature_offsets(self._M, self._radius)

        self._clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=4)


    def generate_dataset(self, ms):

        n_features = len(self._offsets)

        ########
        # Generate the Dataset for training
        ########
        X_lh = []
        y_lh = []

        X_rh = []
        y_rh = []

        X_bg = []
        y_bg = []

        i_gen = ms.img_generator()

        for i, (mask, dmap, fpath) in enumerate(i_gen):

            # print ("Extracting Features from Image:", i)
            start_time = time.time()

            # print("mask shape", mask.shape, fpath)

            lh_results = self.extract_features(mask[:, :, s.LH], dmap, s.LH_LBL, self._offsets, self._n_samples)
            rh_results = self.extract_features(mask[:, :, s.RH], dmap, s.RH_LBL, self._offsets, self._n_samples)

            bg_mask = np.logical_not(mask[:, :, s.LH] + mask[:, :, s.RH])         # combine left and right to find background
            bg_results = self.extract_features(bg_mask, dmap, s.BG_LBL, self._offsets, self._n_samples)

            # print ("LH", np.array(lh_results[0]).shape)
            # print ("RH", np.array(rh_results[0]).shape)

            # check if masks ok
            if np.array(lh_results[0]).shape != (self._n_samples, len(self._offsets)):
                print("Invalid LH Mask in file {}".format(fpath))
                continue

            if np.array(rh_results[0]).shape != (self._n_samples, len(self._offsets)):
                print("Invalid RH Mask in file {}".format(fpath))
                continue

            # append if masks ok
            X_lh.append(lh_results[0])
            y_lh.append(lh_results[1])

            X_rh.append(rh_results[0])
            y_rh.append(rh_results[1])

            X_bg.append(bg_results[0])
            y_bg.append(bg_results[1])

            end_time = time.time()
            total_time = end_time-start_time
            # print ("\tDone: Took %f seconds" % total_time)

        print ("XLH:", np.array(X_lh).shape)
        print ("XRH:", np.array(X_rh).shape)
        print ("XBG:", np.array(X_bg).shape)

        X = np.concatenate((X_lh, X_rh, X_bg))
        y = np.concatenate((y_lh, y_rh, y_bg))

        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        y = y.reshape(y.shape[0]*y.shape[1])

        print ("X:", X.shape)
        print ("y:", y.shape)

        return X, y

    def fit(self, X, y):
        self._clf.fit(X, y)


    def predict(self, X):
        return self._clf.predict(X)


    def generate_mask(self, depth_map):

        mask_shape = (depth_map.shape[0], depth_map.shape[1], 3)
        mask = np.zeros(mask_shape, dtype="uint8")

        all_features1 = dcf.calc_features(depth_map, self._offsets)

        p = self._clf.predict(np.array(all_features1).squeeze())
        p = p.reshape(depth_map.shape)

        mask[:, :, s.LH][p == s.LH_LBL] = 255
        mask[:, :, s.RH][p == s.RH_LBL] = 255

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #
        # mask[:, :, s.LH] = cv2.morphologyEx(mask[:, :, s.LH], cv2.MORPH_OPEN, kernel)
        # mask[:, :, s.RH] = cv2.morphologyEx(mask[:, :, s.RH], cv2.MORPH_OPEN, kernel)

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
        #print ("\tTotal Time for Samples: %f seconds: " % total_time)

        return X, y


if __name__ == "__main__":

    from vpt.streams.compressed_stream import CompressedStream
    from sklearn.metrics import accuracy_score

    s.participant = "mix"
    s.sensor = "realsense"

    training_participants = ["p1", "p2", "p3", "p4", "p6"]
    data_folders = {p : "data/rdf/testing/{}".format(p) for p in training_participants}
    test_folders = {p : "data/rdf/testing/{}".format(p) for p in training_participants}

    for testing_p in training_participants:

        print("#### Testing Participant {} ####".format(testing_p))

        training_folders = [folder for p, folder in data_folders.items() if p != testing_p]
        test_folder = [test_folders[testing_p]]

        cs = CompressedStream(training_folders)

        refresh = False
        M = 5
        radius = .1
        n_samples = 200
        seg_model_path = "data/rdf/trainedmodels/{:s}_M{:d}_rad{:0.2f}".format("mixed_no_{}".format(testing_p), M, radius)

        print(training_folders)
        print(test_folders)
        print(seg_model_path)
        print("M:", M)
        print("Rad:", radius)

        model_p = "mixed_no_{}".format(testing_p)
        rdf_hs = c.load_hs_model(model_p, M, radius, n_samples, refresh=refresh, segmentation_model_path=seg_model_path, ms=cs)

        print("\n## Testing Model...", flush=True)
        cs_test = CompressedStream(test_folder)
        i_gen = cs_test.img_generator()

        avg_accuracy = 0
        total = 0
        for i, (mask, dmap, fpath) in enumerate(i_gen):

            p_mask = rdf_hs.generate_mask(dmap)
            comb = np.vstack((p_mask, mask))

            accuracy = accuracy_score(mask[:,:,:2].ravel(), p_mask[:,:,:2].ravel())
            avg_accuracy += accuracy
            #print ("Accuracy:", accuracy)
            total += 1

            dmap_img = (ip.normalize(dmap)*255).astype('uint8')
            #cv2.imshow("Masks", comb)
            # cv2.imshow("DMap", dmap_img)
            #if cv2.waitKey(1) == ord('q'):
            #   break

        #cv2.destroyAllWindows()
        print ("Avg Accuracy:", avg_accuracy/total)
        print()
        print()
