# David Johnson
#   posture.py
#   Script to generate datasets for use with posture detection training and testing

from vpt.common import *
import vpt.settings as s
from vpt.hand_detection.hand_detector import HandDetector
from vpt.hand_detection.hand_generator import HandGenerator
from vpt.streams.file_stream import FileStream
from vpt.features.features import *

#TODO: Make sure hand generator is smoothing masks to remove noise

PARTICIPANTS = ["p1", "p3", "p4", "p6"]

def generate_dataset(feature_type, hg):

    X_lh = []
    y_lh = []
    X_rh = []
    y_rh = []
    filenames = []

    hgen = hg.hand_generator(debug=False)
    for lh, rh in hgen:
        if lh.label() != None and rh.label() != None:
            filenames.append(lh.get_fpath())

            y_lh.append(lh.label())
            X_lh.append(
                extract_features(lh.get_hand_img(), feature_type))  # TODO: Update with varying parameters for paper

            y_rh.append(rh.label())
            X_rh.append(extract_features(rh.get_hand_img(), feature_type))

    return X_lh, y_lh, X_rh, y_rh, filenames


def generate_datasets(test_participant, M, radius, n_samples, feature_type):

    # load hand generator using RDF Segmenter Model for proper test participant
    seg_model_path = "data/rdf/trainedmodels/{:s}_M{:d}_rad{:0.2f}".format("mixed_no_{}".format(test_participant), M, radius)
    model_name = "mixed_no_{}".format(test_participant)
    rdf_hs = load_hs_model(model_name, M, radius, n_samples, refresh=False, segmentation_model_path=seg_model_path, ms=None)  # stream not needed since we assume model already exists

    # general hand detector used for all participants
    hd = HandDetector(rdf_hs)

    training_folders = {p : "data/posture/{}".format(p) for p in PARTICIPANTS if p != test_participant}
    test_folder = "data/posture/{}".format(test_participant)

    # generate and save  datasets for all training participants
    for participant, folder in training_folders.items():

        annotations = load_annotations(os.path.join(folder, "annotations.txt"), debug=False, error=False)
        fs = FileStream(folder, ftype="bin", annotations=annotations, ignore=True)
        hg = HandGenerator(fs, hd, annotations)

        X_lh, y_lh, X_rh, y_rh, filenames = generate_dataset(feature_type, hg)

        dataset_file = "data{}_test{}_M{}_rad{}_ftype{}.npz".format(participant, test_participant, M, radius, feature_type)
        dataset_path = os.path.join(folder, "datasets")
        os.makedirs(dataset_path, exist_ok=True)
        np.savez_compressed(os.path.join(dataset_path, dataset_file), X_lh=X_lh, y_lh=y_lh, X_rh=X_rh, y_rh=y_rh, filenames=filenames)

    # generate and save dataset for testing participant
    annotations = load_annotations(os.path.join(test_folder, "annotations.txt"), debug=False, error=False)
    fs = FileStream(test_folder, ftype="bin", annotations=annotations, ignore=True)
    hg = HandGenerator(fs, hd, annotations)
    X_lh, y_lh, X_rh, y_rh, filenames = generate_dataset(feature_type, hg)

    dataset_file = "data{}_test{}_M{}_rad{}_ftype{}.npz".format(test_participant, test_participant, M, radius, feature_type)
    dataset_path = os.path.join(test_folder, "datasets")
    os.makedirs(dataset_path, exist_ok=True)
    np.savez_compressed(os.path.join(dataset_path, dataset_file), X_lh=X_lh, y_lh=y_lh, X_rh=X_rh, y_rh=y_rh, filenames=filenames)


def main():

    import vpt.hand_detection.depth_context_features as dcf
    import vpt.hand_detection.depth_image_features as dif

    s.participant = "mix"
    s.sensor = "realsense"

    feature_modules = [dcf, dif]

    Ms = [5]
    radii = [.07]
    feature_types = ["hog"]
    n_samples = 200

    for feature_type in feature_types:
        for test_participant in PARTICIPANTS:
            for M in Ms:
                for radius in radii:
                    generate_datasets(test_participant, M, radius, n_samples, feature_type)