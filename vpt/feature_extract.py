import sys
sys.path.append("./")

import progressbar

import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.streams.hand_stream import *
from vpt.features.features import *


def get_handstream(M, radius, participants):

    basefolder = "data/hands"
    filename = "hands-M{:d}-rad{:0.2f}-p{:s}.npz".format(M, radius, participants)
    return HandStream(os.path.join(basefolder, filename))


def save_data(X_lh, y_lh, X_rh, y_rh, filenames, M, radius, feature_type, data_type, participants):
    base = "data/posture/extracted/"
    data_path = os.path.join(base, "{}_M{}_rad{:0.2f}_{}_".format(participants, M, radius, feature_type))
    np.savez_compressed(data_path + data_type + "_data_combined.npz", X_lh=X_lh, y_lh=y_lh, X_rh=X_rh, y_rh=y_rh, filenames=filenames)

def data_gen(hs, feature_type):

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

            x_lh = extract_features(lh_dmap, feature_type, visualise=False)
            x_rh = extract_features(rh_dmap, feature_type, visualise=False)

            X_lh.append(x_lh)
            y_lh.append(lh_label)

            X_rh.append(x_rh)
            y_rh.append(rh_label)

            filenames.append(filename)

    return np.array(X_lh), np.array(y_lh), np.array(X_rh), np.array(y_rh), np.array(filenames)



def main():


    ## RDF Parameters
    M = 5
    radius = .15
    participants = "all"

    ## Posture Detection Parameters
    feature_type = "test"

    data_type = "train"

    hs = get_handstream(M, radius, participants)

    print("Extracting Features...")
    X_lh, y_lh, X_rh, y_rh, filenames = data_gen(hs, feature_type)

    print("Saving Features...")
    save_data(X_lh, y_lh, X_rh, y_rh, filenames, M, radius, feature_type, data_type=data_type, participants="all_participants")


if __name__ == '__main__':
    main()
