import os
import numpy as np
from vpt.streams.mask_stream import MaskStream
from vpt.streams.mask_stream2 import MaskStream2

import vpt.settings as s

def main(starting_num):

    s.sensor = "realsense"
    new_folder = "data/rdf/mixed"

    folder = "data/rdf/p4"
    ann_masks = MaskStream(folder, ftype=".npy")

    folder2 = "data/rdf/generated/p0"
    gen_masks = MaskStream2(folder2, ftype="mask.npy")

    folder3 = "data/rdf/p1/seq_masks/masks"
    ann_masks2 = MaskStream(folder3, ftype=".npy")


    if os.path.exists(os.path.join(new_folder, "orig_maskpaths.npy")):
        files = np.load(os.path.join(new_folder, "orig_maskpaths.npy")).tolist()
    else:
        files = []

    filenum = starting_num
    for ms in [ann_masks, ann_masks2, gen_masks]:

        m_gen = ms.img_generator()

        for i, (mask, dmap, fpath) in enumerate(m_gen):

            np.save(os.path.join(new_folder, "{:06d}_mask.npy".format(filenum)), mask)
            np.save(os.path.join(new_folder, "{:06d}_orig.npy".format(filenum)), dmap)

            files.append(fpath)
            filenum+=1

    np.save(os.path.join(new_folder, "orig_maskpaths.npy"), files)

if __name__ == "__main__":

    main(0)