import unittest
from sklearn.svm import LinearSVC

from vpt.hand_detection.hand_detector import *
from vpt.streams.file_stream import *
from vpt.hand_detection.hand_generator import *
from vpt.hand_detection.rdf_segmentation_model import *

from vpt.features.features import *
import vpt.settings as s
from vpt.common import *

# TODO:  Expand to include some classification results...

class ClfTester(unittest.TestCase):

    def setUp(self):

        s.participant = "p0"
        s.sensor = "kinect"
        M = 3
        radius = .3
        n_samples = 250
        refresh = False
        segmentation_model_path = "%s_M%i_rad%0.1f" % (s.participant if s.participant != "p9" else "p1", M, radius)  # added p9 check for testing with smaller dataset

        test_folder = "data/testdata/p0/"
        ftype = "bin"

        annotation_file = "data/testdata/p0/annotations.txt"
        annotations = load_annotations(annotation_file)

        fs = FileStream(test_folder, ftype)
        hs = load_hs_model(s.participant, M, radius, n_samples, refresh, segmentation_model_path)
        hd = HandDetector(hs)
        self._hg = HandGenerator(fs, hd, annotations)

        self._nfiles = 844
        self._n_hogFeatures = 968
        self._n_shogFeatures = 160

    def test_hog(self):

        lhs = []
        rhs = []

        for i, (lh, rh) in enumerate(self._hg.hand_generator(debug=True)):

            if i == 20:
                break

            lhs.append(lh)
            rhs.append(rh)

        X_lh, y_lh = generate_data_set(lhs, "hog", training=True)
        self.assertEqual(X_lh.shape[0], y_lh.shape[0], "Number of labels and features vectors don't match")
        self.assertEqual(X_lh.shape[1], self._n_hogFeatures, "Incorrect number of HOG features")

        X_rh, y_rh = generate_data_set(rhs, "hog", training=True)
        self.assertEqual(X_rh.shape[0], y_rh.shape[0], "Number of labels and features vectors don't match")
        self.assertEqual(X_rh.shape[1], self._n_hogFeatures, "Incorrect number of HOG features")

        print "LH X Shape:", X_lh.shape
        print "RH X Shape:", X_rh.shape


    def test_shog(self):

        lhs = []
        rhs = []

        for i, (lh, rh) in enumerate(self._hg.hand_generator()):

            if i == 20:
                break

            lhs.append(lh)
            rhs.append(rh)

        X_lh, y_lh = generate_data_set(lhs, "shog", training=True)
        self.assertEqual(X_lh.shape[0], y_lh.shape[0], "Number of labels and features vectors don't match")
        self.assertEqual(X_lh.shape[1], self._n_shogFeatures, "Incorrect number of SHOG features")

        X_rh, y_rh = generate_data_set(rhs, "shog", training=True)
        self.assertEqual(X_rh.shape[0], y_rh.shape[0], "Number of labels and features vectors don't match")
        self.assertEqual(X_rh.shape[1], self._n_shogFeatures, "Incorrect number of SHOG features")

        print "LH X Shape:", X_lh.shape
        print "RH X Shape:", X_rh.shape


if __name__ == "__main__":

    unittest.main()