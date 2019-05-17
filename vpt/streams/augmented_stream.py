from vpt.common import *
from vpt.settings import *


class AugmentedStream:

    def __init__(self, folders, ftype="npz", annotations=None, normalize=False, ignore=False):

        self._folders = folders
        self._ftype = ftype
        self._fpaths = []
        self._annotations = annotations
        self._normalize = normalize
        self._ignore = ignore
        self._strip = annotations != None

        self._og_fpaths = np.zeros((0,), dtype=str)

        self.load_filenames()



    def load_filenames(self):
        for folder in self._folders:
            og_fpaths = np.load(os.path.join(folder, "filenames.npy"))
            self._og_fpaths = np.hstack((self._og_fpaths, og_fpaths))

        idx = 0
        for folder in self._folders:
            for root, dirs, files in os.walk(folder, followlinks=True):
                for fname in files:
                    if self._ftype in fname and "background" not in root:  # exclude folders with background in name
                        fpath = os.path.join(root, fname)
                        fpath_og = self._og_fpaths[idx]

                        if self._strip:
                            try:
                                labels = (None, None)
                                if self._annotations != None:
                                    key = getFileKey(fpath_og)
                                    labels = self._annotations[key]
                                    if self._ignore:
                                        if labels[0] in ANNOTATIONS and labels[1] in ANNOTATIONS:
                                            self._fpaths.append(fpath)
                                            idx += 1
                                    else:
                                        self._fpaths.append(fpath)
                                        idx += 1

                            except Exception as e:
                                print (e)
                        else:
                            self._fpaths.append(fpath)
                            idx += 1

        print ("# Files Loaded:", len(self._fpaths))
        print ("# OG File Names:", len(self._og_fpaths))


    def img_generator(self):

        for fname, og_fname in zip(self._fpaths, self._og_fpaths):

            try:
                hands = np.load(fname)
                dmap = hands["dmap"]
                dmap = background_sub(dmap)
                yield dmap, og_fname

            except Exception as e:
                print ("Error Generating Image:", e, fname)


    def get_fpaths(self):
        print ("Shape:", len(self._fpaths))
        return self._fpaths



def test():

    folders = [ "data/rdf/training-rand/p6", "data/rdf/training-rand/p3"]
    aus = AugmentedStream(folders)

    mgen = aus.img_generator()
    for dmap, fname in mgen:

        dmap_img = (ip.normalize(dmap)*255).astype('uint8')
        cv2.imshow("Dmap", dmap_img)
        print(fname)
        if cv2.waitKey(300) == ord('q'):
            break


if __name__ == '__main__':
    test()