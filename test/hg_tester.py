import unittest
from vpt.common import *

from vpt.streams.file_stream import *
from vpt.hand_detection.hand_generator import *

class HandGenTester(unittest.TestCase):

    def setUp(self):
        self._test_folder = "data/testdata/p0/"
        self._ftype = "bin"
        self._img_test = np.load("data/testdata/filestream_test.npy")
        self._nfiles = 844

        self._fs = FileStream(self._test_folder, self._ftype)

        s.participant = "p0"
        M = 3
        radius = .3
        n_samples = 250
        refresh = False
        segmentation_model_path = "%s_M%i_rad%0.1f" % (s.participant if s.participant != "p9" else "p1", M, radius)  # added p9 check for testing with smaller dataset

        hs = load_hs_model(s.participant, M, radius, n_samples, refresh, segmentation_model_path)
        self.hd = HandDetector(hs)
        hg = HandGenerator(self._fs, self.hd)
        self._hand_gen = hg.hand_generator()



    def test_filestream(self):

        self.assertEqual(len(self._fs._fpaths), self._nfiles)

        img_gen = self._fs.img_generator()

        i = 0
        for i, (img, fpath) in enumerate(img_gen):

            if fpath == "data/testdata/p0/error_2/p0e/002929.bin":
                # numbers are beginning and test file numbers
                self.assertTrue(np.array_equal(img, self._img_test))

        self.assertEqual(i, self._nfiles-1)


    def test_handgenerator(self):

        for i, (lh, rh) in enumerate(self._hand_gen):
            self.assertIsNotNone(lh)
            self.assertIsNotNone(rh)

            if i == 10:
                break


    def test_handgenerator_with_annotations(self):

        annnotation_file = "data/testdata/p0/annotations.txt"
        annotations = load_annotations(annnotation_file, error=True)

        key = getFileKey("/Users/fortjay81/Dropbox/Uvic/dissertation/vpt/src/data/p0/error_1/p0d/001421.bin")
        a = annotations[key]
        self.assertTupleEqual(('1', '1'), tuple(a))

        key = getFileKey("/Users/fortjay81/Dropbox/Uvic/dissertation/vpt/src/data/p0/error_0/p0e/002522.bin")
        a = annotations[key]
        self.assertTupleEqual(('0', '0'), tuple(a))

        hg = HandGenerator(self._fs, self.hd, annotations)
        hand_gen = hg.hand_generator()


        for i, (lh, rh) in enumerate(hand_gen):
            self.assertIsNotNone(lh)
            self.assertIsNotNone(rh)

            self.assertEqual(lh.label(), 2)
            self.assertEqual(rh.label(), 2)

            if i == 10:
                break


    def test_sub_handgenerator(self):

        hs = SubSegmentationModel("TestModel")

        background_folder = "data/testdata/p0/background_model_2"
        hs = SubSegmentationModel("Test")
        hs.initialize(background_folder, history=50, varThreshold=.3, shadowDetection=False)

        hd = HandDetector(hs)
        hg = HandGenerator(self._fs, hd)

        for i, (lh, rh) in enumerate(hg.hand_generator()):
            self.assertIsNotNone(lh)
            self.assertIsNotNone(rh)

            if i == 10:
                break



if __name__ == '__main__':

    unittest.main()