import sys
sys.path.append("./")

import progressbar

import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.streams.hand_stream import *
from vpt.features.features import *
from vpt.hand_detection.hand_generator import *
import vpt.settings as s

import vpt.individual_eval as eval

def save_data(X_lh, y_lh, X_rh, y_rh, filenames, M, radius, feature_type, data_type, participants):
    base = "data/posture/extracted/"
    data_path = os.path.join(base, "{}_M{}_rad{:0.2f}_{}_".format(participants, M, radius, feature_type))
    np.savez(data_path + data_type + "_data_combined.npz", X_lh=X_lh, y_lh=y_lh, X_rh=X_rh, y_rh=y_rh, filenames=filenames)

def get_handstream(M, radius, participants):

    basefolder = "data/hands"
    filename = "hands-M{:d}-rad{:0.2f}-p{:s}.npz".format(M, radius, participants)
    return HandStream(os.path.join(basefolder, filename))


def data_gen(hs, feature_func, pixels_per_cell=(8,8), cells_per_block=(3,3), img_size=(128, 128)):

    X_lh = []
    y_lh = []

    X_rh = []
    y_rh = []

    filenames = []

    hgen = hs.hand_generator()
    bar = progressbar.ProgressBar(max_value=15817)
    for i, (lh_dmap, lh_label, rh_dmap, rh_label, filename) in bar(enumerate(hgen)):

            rh_dmap = rh_dmap[:, ::-1]

            # TODO: Add smoothing??

            x_lh = feature_func(lh_dmap, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm="L1-sqrt", img_size=img_size, pad_img=False)
            x_rh = feature_func(rh_dmap, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm="L1-sqrt", img_size=img_size, pad_img=False)

            X_lh.append(x_lh)
            y_lh.append(lh_label)

            X_rh.append(x_rh)
            y_rh.append(rh_label)

            filenames.append(filename)

    return np.array(X_lh), np.array(y_lh), np.array(X_rh), np.array(y_rh), np.array(filenames)


def main():

    features = [hog]
    cell_sizes = [(4,4), (6,6), (8,8), (12,12), (16,16)]
    block_sizes = [(3,3), (4,4)]

    for f in features:
        for c in cell_sizes:
            for b in block_sizes:

                ## RDF Parameters
                M = 5
                radius = .15
                participants = "all"

                if f == honv:
                    fname = "honv"
                else:
                    fname = "hog"
                feature_type = "f_{}-c_{}-b_{}".format(fname, c[0], b[0])

                data_type = "train"

                hs = get_handstream(M, radius, participants)
                print("Extracting Features:", fname, "- Pixels per Cell:", c, "- Cells per Block:", b)
                X_lh, y_lh, X_rh, y_rh, filenames = data_gen(hs, f, pixels_per_cell=c, cells_per_block=b)
                print("Saving Data...")
                save_data(X_lh, y_lh, X_rh, y_rh, filenames, M, radius, feature_type, data_type=data_type,  participants="all_participants")


if __name__ == '__main__':
    main()