from vpt.hand_detection.hand_detector import *
from vpt.streams.file_stream import *
from vpt.common import  *


class HandGenerator():

    def __init__(self, stream, detector, annotations=None):

        self._stream = stream
        self._detector = detector
        self._img_gen = self._stream.img_generator()
        self._annotations = annotations


    def hand_generator(self, debug=False):

        for img, fpath in self._img_gen:
            try:
                labels = (None, None)
                if self._annotations != None:
                    key = getFileKey(fpath)
                    labels = self._annotations[key]

                yield self._detector.detect_hands(img, fpath, labels)
            except ValueError as e:
                print "Error in Hand Generator:", e.message
            except Exception as e:
                if debug:
                    print "Error in Hand Generator:", e