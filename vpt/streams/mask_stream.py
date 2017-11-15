from vpt.common import *
from vpt.settings import *

class MaskStream:

    def __init__(self, folder, ftype="npy", annotations=None, normalize=False, ignore=False):

        self._folder = folder
        self._ftype = ftype
        self._fpaths = []
        self._annotations = annotations
        self._normalize = normalize
        self._ignore = ignore

        self._strip = annotations != None

        self.load_filenames()


    def load_filenames(self):
        for root, dirs, files in os.walk(self._folder):
            for fname in files:
                if self._ftype in fname and "background" not in root:  # exclude folders with background in name
                    fpath = os.path.join(root, fname)

                    if self._strip:
                        try:
                            labels = (None, None)
                            if self._annotations != None:
                                key = getFileKey(fpath)
                                labels = self._annotations[key]
                                if self._ignore:
                                    if labels[0] in ANNOTATIONS and labels[1] in ANNOTATIONS:
                                        self._fpaths.append(fpath)
                                else:
                                    self._fpaths.append(fpath)
                        except Exception as e:
                            print (e)
                    else:
                        self._fpaths.append(fpath)

        print ("# Files Loaded:", len(self._fpaths))


    def img_generator(self):

        og_folder = "data/posture"

        for fname in self._fpaths:

            # assumes masks are in rdf folder...TODO: Should probably change to Regex
            try:
                temp = fname.split("/")
                participant = temp[2]
                exercise = temp[5]
                file_num = temp[-1][:6]

                og_path = os.path.join(og_folder, participant, exercise, file_num + ".bin")
            except IndexError as e:
                raise ValueError("Error: Mask Stream initialized with a non mask folder", e)

            try:
                yield load_depthmap(fname, normalize=False), load_depthmap(og_path, normalize=self._normalize), fname
            except Exception as e:
                print ("Error Generating Image:", e)


    def get_fpaths(self):
        print ("Shape:", len(self._fpaths))
        return self._fpaths