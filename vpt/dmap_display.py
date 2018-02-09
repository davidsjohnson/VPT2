import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.streams.file_stream import *
from vpt.hand_detection.hand_generator import *
from vpt.hand_detection.hand_detector import *
from vpt.features.features import *

import matplotlib.animation as animation

def data_gen():

    ## Some General Parameters
    s.participant = "all"
    s.sensor = "realsense"

    participants = ["p6"]
    posture_folders = {p: os.path.join("data/posture", p) for p in participants}
    ftype = ".bin"

    # annotation_file = "data/posture/annotations.txt"
    # annotations = load_annotations(annotation_file)

    #### Generate and Save data for all testing participants
    folders = [folder for p, folder in posture_folders.items()]

    fs = FileStream(folders, ftype)



    igen = fs.img_generator()
    for i, (dmap, fname) in enumerate(igen):
        yield dmap, fname


def updatefig(data):
    dmap, fname = data

    axes.set_title(("File: {}".format(fname)))
    im.set_array(dmap)

    return im,


rows = 1
cols = 1

fig, axes = plt.subplots(rows, cols)
im = plt.imshow(np.zeros((192,480)), animated=True, vmin=0, vmax=1000)

axes.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", labelleft="off", left="off")

ani = animation.FuncAnimation(fig, updatefig, data_gen, interval=1, blit=True, repeat=False)
plt.show()
