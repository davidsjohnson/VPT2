import unittest
from vpt.common import *
import vpt.settings as s

from vpt.hand_detection.hand_detector import *


class RDFHandDetectorTester(unittest.TestCase):

    def setUp(self):

        self.display = False

        s.participant = "p0"
        M = 3
        radius = .3
        n_samples = 250
        refresh = False
        segmentation_model_path = "%s_M%i_rad%0.1f" % (s.participant if s.participant != "p9" else "p1", M, radius)  # added p9 check for testing with smaller dataset

        hs = load_hs_model(s.participant, M, radius, n_samples, refresh, segmentation_model_path)
        self.hd = HandDetector(hs)

        f_depth = "data/testdata/hs/annotations/p0d/001629_orig.npy"
        self.depthmap = np.load(f_depth)


    def test_detector(self):

        lh, rh = self.hd.detect_hands(self.depthmap)
        self.assertIsNotNone(lh)
        self.assertIsNotNone(rh)

        lh_img = lh.get_hand_img()
        rh_img = rh.get_hand_img()

        lh_truth = np.load("data/testdata/left_hand_img.npy")
        rh_truth = np.load("data/testdata/rght_hand_img.npy")

        if self.display:
            plt.figure()
            plt.subplot(231)
            plt.imshow(self.depthmap, cmap="gray")
            plt.subplot(232)
            plt.imshow(lh_img, cmap="gray")
            plt.subplot(233)
            plt.imshow(lh_truth, cmap="gray")
            plt.subplot(234)
            plt.imshow(self.depthmap, cmap="gray")
            plt.subplot(235)
            plt.imshow(rh_img, cmap="gray")
            plt.subplot(236)
            plt.imshow(rh_truth, cmap="gray")
            plt.show()

        self.assertTrue(np.array_equal(lh_img, lh_truth))
        self.assertTrue(np.array_equal(rh_img, rh_truth))


class SubHandDetectorTester(unittest.TestCase):

    def setUp(self):

        self.display = False

        background_folder = "data/testdata/p0/background_model_2"
        hs = SubSegmentationModel("Test")
        hs.initialize(background_folder, history=50, varThreshold=.3, shadowDetection=False)

        self.hd = HandDetector(hs)

        f_depth = "data/testdata/p0/error_2/p0e/002336.bin"
        self.depthmap = load_depthmap(f_depth)


    def test_sub_detector(self):
        lh, rh = self.hd.detect_hands(self.depthmap)
        self.assertIsNotNone(lh)
        self.assertIsNotNone(rh)

        lh_img = lh.get_hand_img()
        rh_img = rh.get_hand_img()

        lh_truth = np.load("data/testdata/bgsub_lh_test.npy")
        rh_truth = np.load("data/testdata/bgsub_rh_test.npy")

        if self.display:
            plt.figure()
            plt.subplot(231)
            plt.imshow(self.depthmap, cmap="gray")
            plt.subplot(232)
            plt.imshow(lh_img, cmap="gray")
            plt.subplot(233)
            plt.imshow(lh_truth, cmap="gray")
            plt.subplot(234)
            plt.imshow(self.depthmap, cmap="gray")
            plt.subplot(235)
            plt.imshow(rh_img, cmap="gray")
            plt.subplot(236)
            plt.imshow(rh_truth, cmap="gray")
            plt.show()

        self.assertTrue(np.array_equal(lh_img, lh_truth))
        self.assertTrue(np.array_equal(rh_img, rh_truth))



if __name__ == "__main__":

    unittest.main()