import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.streams.file_stream import *
from vpt.hand_detection.hand_generator import *
from vpt.hand_detection.hand_detector import *
from vpt.features.features import *

import matplotlib.animation as animation


X_lh = []
y_lh = []
vis_lhs = []

X_rh = []
y_rh = []
vis_rhs = []

filenames = []

## RDF Parameters
M = 5
radius = .15

## Posture Detection Parameters
feature_type = "shog"

def save_data(X_lh, y_lh, X_rh, y_rh, filenames, M, radius, feature_type, data_type):
    base = "data/posture/extracted/"

    data_path = os.path.join(base, "{}_M{}_rad{:0.2f}_{}_".format("all_participants2-noblock", M, radius, feature_type))
    np.savez_compressed(data_path + data_type + "_data.npz", X_lh=X_lh, y_lh=y_lh, X_rh=X_rh, y_rh=y_rh,
                        filenames=filenames)


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


def data_gen():

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

    print(folders)

    hg = init_hg(folders, "all", annotation_file, offset_gen, feature_gen,
                             M, radius, n_samples=750, ftype=".bin")

    hgen = hg.hand_generator(debug=True)
    for i, (lh, rh) in enumerate(hgen):

        if lh.label() != None and rh.label() != None:
            x_lh, vis_lh = extract_features(lh.get_hand_img(), feature_type, visualise=True)
            x_rh, vis_rh = extract_features(rh.get_hand_img(), feature_type, visualise=True)

            X_lh.append(x_lh)
            y_lh.append(lh.label())
            vis_lhs.append(vis_lh)

            X_rh.append(x_rh)
            y_rh.append(rh.label())
            vis_rhs.append(vis_rh)

            filenames.append(lh.get_fpath())

            yield lh, vis_lh, rh, vis_rh

        else:
            raise RuntimeWarning("Warning: No label found for hands")


def updatefig(data):
    lh, vis_lh, rh, vis_rh = data

    axes[0][0].set_title("LH | {}".format(lh.get_fpath()))
    axes[0][1].set_title("RH | {}".format(rh.get_fpath()))

    axes[2][0].set_title("Label {}".format(lh.label()))
    axes[2][1].set_title("Label {}".format(rh.label()))

    for i, im in enumerate(ims):
        if i//2 == 0:
            if i%2 == 0:
                im.set_array(lh.get_original())
            else:
                im.set_array(rh.get_original())
        elif i//2 == 1:
            if i%2 == 0:
                im.set_array(lh.get_mask())
            else:
                im.set_array(rh.get_mask())
        elif i//2 == 2:
            if i%2 == 0:
                im.set_array(lh.get_hand_img())
            else:
                im.set_array(rh.get_hand_img())
        else:
            if i%2 == 0:
                im.set_array(vis_lh)
            else:
                im.set_array(vis_rh)

    return ims


rows = 4
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(8,8))
ims = [axes[i][j].imshow(np.zeros((192,480)), animated=True, vmin=0, vmax=850)      if i == 0
       else axes[i][j].imshow(np.zeros((192,480)), animated=True, vmin=0, vmax=255) if i == 1
       else axes[i][j].imshow(np.zeros((120,90)), animated=True, vmin=0, vmax=850)  if i == 2
       else axes[i][j].imshow(np.zeros((120,90)), animated=True, vmin=0, vmax=.00085)
       for i in range(rows) for j in range(cols)]
[ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", labelleft="off", left="off") for row in axes for ax in row]

ani = animation.FuncAnimation(fig, updatefig, data_gen, interval=1, blit=True, repeat=False)
plt.show()

print("Saving feature vis")

X_lh = np.array(X_lh)
y_lh = np.array(y_lh)
X_rh = np.array(X_rh)
y_rh = np.array(y_rh)
filenames = np.array(filenames)

save_data(X_lh, y_lh, X_rh, y_rh, filenames, M, radius, feature_type, data_type="train")
np.savez_compressed("data/posture/extracted/all_feature_vis-noblock.npz", vis_lhs=vis_lhs, y_lh=y_lh, vis_rhs=vis_rhs, y_rh=y_rh, filenames=filenames)