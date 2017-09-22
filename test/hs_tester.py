## Unit Testing ##
import unittest

from sklearn.metrics import *
#
# from vpt.hand_detection.rdf_segmentation_model import *
from vpt.hand_detection.rdf_segmentation_model_old import *
from vpt.hand_detection.sub_segmentation_model import *
from vpt.common import *
import vpt.hand_detection.depth_context_features as dcf

import vpt.settings as s


class DCFTester(unittest.TestCase):

    def setUp(self):

        s.participant = "p0"
        s.sensor      = "kinect"

        self.display = False

        self.M = 3
        self.radius = .3
        self.offsets = np.array(dcf.generate_feature_offsets(self.M, self.radius))
        self.offset_len = (self.M * 2 + 1) * (self.M * 2 + 1) - 1

        self.depthmap = load_depthmap("data/testdata/p0/error_2/p0e/002400.bin")

        # Pixel Values
        self.u = 250
        self.v = 120
        self.depth = self.depthmap[self.v,self.u]          # value = 849

        self.sample_mask = np.zeros_like(self.depthmap, dtype=bool)
        self.sample_mask[self.v,self.u] = True

        self.point = dcf.pixels2points(self.depthmap, self.sample_mask)


    def test_offsets(self):


        self.assertEqual(self.offsets.shape, (self.offset_len,3))
        self.assertEqual(self.offsets.max(), self.radius)
        self.assertEqual(self.offsets.min(), -self.radius)

        if self.display:
            plt.scatter(self.offsets[:, 0], self.offsets[:,1])
            plt.show()

    def test_pixels2points(self):

        self.assertEqual(self.point.shape, (1,3))

        #calculate rw values
        f = 525.5
        px_d = 320.0
        py_d = 240.0

        z = self.depth * .001 + .00001       #convert to meters and add non zero term
        x = (self.u - px_d) * z / f
        y = (self.v - py_d) * z / f

        point_truth = np.array([x, y, z])

        print "Point", self.point
        print "Truth", point_truth

        self.assertTrue(np.allclose(point_truth, self.point[0], rtol=1e-04, atol=1e-07))


    def test_points2pixels(self):

        self.point = np.expand_dims(self.point, 1)      # necessary for feature calc
        pixel = dcf.points2pixels(self.point)

        print "Calc Pixel", pixel

        self.assertEqual(pixel.shape, (1, 1, 2))
        self.assertEqual(pixel[0, 0, 0], self.u)
        self.assertEqual(pixel[0, 0, 1], self.v)


    def test_featurecalc(self):


        features = dcf.calc_features(self.depthmap, self.offsets, self.sample_mask)
        np.save("data/testdata/features_250-120.npy", features)
        feature_truth = np.load("data/testdata/features_250-120.npy")

        # plt.figure()
        # plt.subplot(121)
        # plt.hist(features.ravel())
        # plt.subplot(122)
        # plt.hist(feature_truth.ravel())
        # plt.show()

        if self.display:
            plt.figure()
            plt.subplot(121)
            plt.scatter(np.arange(features.shape[1]), features[0])
            plt.subplot(122)
            plt.scatter(np.arange(feature_truth.shape[1]), feature_truth[0])
            plt.show()

        self.assertEqual(features.shape, (1, self.offset_len))
        self.assertTrue(np.array_equal(features, feature_truth))



class RDFModelTester(unittest.TestCase):

    def setUp(self):

        s.participant = "p0"
        s.sensor = "kinect"
        self.M = 3
        self.radius = .3
        n_samples = 500
        refresh = False
        segmentation_model_path = "%s_M%i_rad%0.1f" % (s.participant if s.participant != "p9" else "p1", self.M, self.radius)  # added p9 check for testing with smaller dataset

        self.hs = load_hs_model(s.participant, self.M, self.radius, n_samples, refresh, segmentation_model_path)
        self.hs2 = RDFSegmentationModelOLD("p0_M3_rad0.3")


    def test_load_model(self):

        self.assertEqual(self.hs._M, self.M)
        self.assertEqual(self.hs._radius, self.radius)
        self.assertNotEqual(self.hs._clf, None)

        self.assertEqual(len(self.hs._offsets), 48)


    def test_prediction(self):


        f_depth = "data/rdf/p9/p9e/000360_orig.npy"
        f_mask = "data/rdf/p9/p9e/000360_mask.npy"
        depth = np.load(f_depth)
        mask = np.load(f_mask)

        p_mask = self.hs.generate_mask(depth)
        p_mask2 = self.hs2.generate_mask(depth)
        score = accuracy_score(mask.ravel(), p_mask2.ravel())

        np.save("data/testdata/p_mask_test.npy", p_mask)

        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(mask)
        # plt.subplot(132)
        # plt.imshow(p_mask)
        # plt.subplot(133)
        # plt.imshow(p_mask2)
        # plt.show()

        p_mask_truth = np.load("data/testdata/p_mask_test.npy")
        self.assertTrue(np.array_equal(p_mask, p_mask_truth))
        self.assertAlmostEqual(score, 0.99,places=2)


class SubModelTester(unittest.TestCase):


    def setUp(self):

        background_folder = "data/testdata/p0/background_model_2"
        self.bg = SubSegmentationModel("TestModel")
        self.bg.initialize(background_folder, history=50, varThreshold=.3, shadowDetection=False)


    def test_prediction(self):

        import matplotlib.pyplot as plt

        f_depth = "data/testdata/p0/error_2/p0e/002336.bin"
        f_mask = "data/testdata/backgroundSub_test.npy"
        depth = load_depthmap(f_depth)
        mask = np.load(f_mask)

        p_mask = self.bg.generate_mask(depth)

        self.assertTrue(np.array_equal(p_mask, mask))


if __name__ == '__main__':

    unittest.main()