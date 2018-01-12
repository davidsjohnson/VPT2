from vpt.common import *
from vpt.settings import *

class CompressedStream:

    def __init__(self, folders, ftype="npz", annotations=None, normalize=False, ignore=False):

        self._folders = folders
        self._ftype = ftype
        self._fpaths = []
        self._annotations = annotations
        self._normalize = normalize
        self._ignore = ignore

        self._strip = annotations != None

        self.load_filenames()


    def load_filenames(self):
        for folder in self._folders:
            for root, dirs, files in os.walk(folder, followlinks=True):
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

        for fname in self._fpaths:

            try:
                hands = np.load(fname)
                mask = hands["mask"]
                dmap = hands["dmap"]
                dmap = background_sub(dmap)

                yield mask, dmap, fname
            except Exception as e:
                print ("Error Generating Image:", e, fname)


    def get_fpaths(self):
        print ("Shape:", len(self._fpaths))
        return self._fpaths



def test():

    folders = [ "data/rdf/training/p6", "data/rdf/training/p3"]
    cs = CompressedStream(folders)

    mgen = cs.img_generator()
    for mask, dmap, fname in mgen:

        dmap_img = (ip.normalize(dmap)*255).astype('uint8')

        lh_img = cv2.bitwise_and(dmap_img, dmap_img, mask=mask[:, :, s.LH])
        rh_img = cv2.bitwise_and(dmap_img, dmap_img, mask=mask[:, :, s.RH])

        cv2.imshow("Mask", mask)
        cv2.imshow("Dmap", dmap_img)
        cv2.imshow("LH", lh_img)
        cv2.imshow("RH", rh_img)


        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    test()