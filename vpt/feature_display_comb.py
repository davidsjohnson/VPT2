import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.streams.hand_stream import *
from vpt.features.features import *

import matplotlib.animation as animation

def get_handstream(M, radius, participants):

    basefolder = "data/hands"
    filename = "hands-M{:d}-rad{:0.2f}-p{:s}.npz".format(M, radius, participants)
    return HandStream(os.path.join(basefolder, filename))


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
participants = "all"

## Posture Detection Parameters
feature_type = "hog3"

data_type = "train"

hs = get_handstream(M, radius, participants)

def save_data(X_lh, y_lh, vis_lhs, X_rh, y_rh, vis_rhs, filenames, M, radius, feature_type, data_type, participants):
    base = "data/posture/extracted/"

    data_path = os.path.join(base, "{}_M{}_rad{:0.2f}_{}_".format(participants, M, radius, feature_type))
    np.savez_compressed(data_path + data_type + "_data_combined.npz", X_lh=X_lh, y_lh=y_lh, vis_lhs=vis_lhs,
                        X_rh=X_rh, y_rh=y_rh, vis_rhs=vis_rhs, filenames=filenames)


def data_gen():

    hgen = hs.hand_generator()
    for i, (lh_dmap, lh_label, rh_dmap, rh_label, filename) in enumerate(hgen):

            rh_dmap = rh_dmap[:, ::-1]

            x_lh, vis_lh = extract_features(lh_dmap, feature_type, visualise=True)
            x_rh, vis_rh = extract_features(rh_dmap, feature_type, visualise=True)

            X_lh.append(x_lh)
            y_lh.append(lh_label)
            vis_lhs.append(vis_lh)

            X_rh.append(x_rh)
            y_rh.append(rh_label)
            vis_rhs.append(vis_rh)

            filenames.append(filename)

            yield lh_dmap, lh_label, vis_lh, rh_dmap, rh_label, vis_rh, filename


def updatefig(data):
    lh_dmap, lh_label, vis_lh, rh_dmap, rh_label, vis_rh, filename = data

    axes[0][0].set_title("LH | {}".format(filename))
    axes[0][1].set_title("RH | {}".format(filename))

    axes[1][0].set_title("Label {}".format(lh_label))
    axes[1][1].set_title("Label {}".format(lh_label))

    for i, im in enumerate(ims):
        if i//2 == 0:
            if i%2 == 0:
                im.set_array(lh_dmap)
            else:
                im.set_array(rh_dmap)
        else:
            if i%2 == 0:
                im.set_array(vis_lh)
            else:
                im.set_array(vis_rh)

    return ims


rows = 2
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(5,5))
ims = [axes[i][j].imshow(np.zeros((180,120)), animated=True, vmin=0, vmax=850)  if i == 0
       else axes[i][j].imshow(np.zeros((180,120)), animated=True, vmin=0, vmax=.00085)
       for i in range(rows) for j in range(cols)]
[ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", labelleft="off", left="off") for row in axes for ax in row]

ani = animation.FuncAnimation(fig, updatefig, data_gen, interval=1, blit=True, repeat=False)
plt.show()

print("Saving Features")

X_lh = np.array(X_lh)
y_lh = np.array(y_lh)
X_rh = np.array(X_rh)
y_rh = np.array(y_rh)
filenames = np.array(filenames)

save_data(X_lh, y_lh, vis_lhs, X_rh, y_rh, vis_rhs, filenames, M, radius, feature_type, data_type=data_type, participants="all_participants")