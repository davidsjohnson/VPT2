import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.streams.file_stream import *
from vpt.hand_detection.hand_generator import *
from vpt.hand_detection.hand_detector import *
from vpt.features.features import *



def init_hg(folders, testing_p, annotation_file, offset_gen, feature_gen,
                        M, radius, n_samples=750, ftype=".bin"):

    annotations = load_annotations(annotation_file)
    fs = FileStream(folders, ftype, annotations=annotations, ignore=True)

    # generate or load model
    base_model_folder = "data/rdf/trainedmodels/"
    seg_model_path = os.path.join(base_model_folder,
                                  "{:s}_M{:d}_rad{:0.2f}".format("mixed_all_participants", M, radius))
    rdf_hs = load_hs_model("RDF Model", offset_gen, feature_gen, M, radius, n_samples, refresh=False,
                           segmentation_model_path=seg_model_path)

    hd = HandDetector(rdf_hs)
    hg = HandGenerator(fs, hd, annotations)

    return hg


def extract_hands(hg):

    lh_dmaps = []
    y_lh = []
    rh_dmaps = []
    y_rh = []
    filenames = []

    hgen = hg.hand_generator(debug=True)
    for i, (lh, rh) in enumerate(hgen):

        if lh.label() != None and rh.label() != None:
            lh_dmaps.append(lh.get_hand_img().copy())
            y_lh.append(lh.label())

            rh_dmaps.append(rh.get_hand_img().copy())
            y_rh.append(rh.label())

            filenames.append(lh.get_fpath())
            print(i,end=", ", flush=True)

        else:
            raise RuntimeWarning("Warning: No label found for hands")


    return np.array(lh_dmaps), np.array(y_lh), np.array(rh_dmaps), np.array(y_rh), np.array(filenames)


if __name__ == '__main__':
    ## RDF Parameters
    M = 5
    radius = .15

    ## Some General Parameters
    s.participant = "all"
    s.sensor = "realsense"

    participants = ["p1", "p3", "p4", "p6"]
    posture_folders = {p: os.path.join("data/posture", p) for p in participants}

    annotation_file = "data/posture/annotations.txt"

    offset_gen = dcf.generate_feature_offsets
    feature_gen = dcf.calc_features

    #### Generate and Save data for all testing participants
    folders = [folder for p, folder in posture_folders.items()]

    hg = init_hg(folders, "all", annotation_file, offset_gen, feature_gen, M, radius, n_samples=750, ftype=".bin")
    print("Extracting {:d}  Hands...".format(hg.size()))
    lh_dmaps, y_lh, rh_dmaps, y_rh, filenames = extract_hands(hg)
    print("Done")
    print("Saving Hands...")
    np.savez_compressed("data/hands/hands-M{:d}-rad{:0.2f}-p{:s}".format(M, radius, s.participant), lh_dmaps=lh_dmaps, y_lh=y_lh, rh_dmaps=rh_dmaps, y_rh=y_rh, filenames=filenames )
    print("Done")