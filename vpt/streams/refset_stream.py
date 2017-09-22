from vpt.common import *
from vpt.streams.file_stream import *

class RefSetStream(FileStream):

    def __init__(self, fpaths, ftype="bin", annotations=None):

        self._fpaths = fpaths
        self._ftype = ftype
        self._annotations = annotations

        self.load_filenames()


    def load_filenames(self):
        print "# Files Loaded:", self._fpaths.shape


    def img_generator(self):
        for fname in self._fpaths:
            try:
                yield load_depthmap(fname), fname
            except Exception as e:
                print e


    def get_fpaths(self):
        print "Shape:", len(self._fpaths)
        return self._fpaths