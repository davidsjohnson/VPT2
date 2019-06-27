## Visualize each component of the 
import sys
sys.path.append("./")
import matplotlib.patches as patches

import vpt.settings as s
import vpt.hand_detection.depth_context_features as dcf
from vpt.common import *
from vpt.hand_detection.hand_detector import *
from vpt.features.features import *

import vpt.utils.image_processing as ip

import matplotlib as mpl
mpl.rc('image', cmap='Spectral')


def get_hd(offset_gen, feature_gen, M, radius, n_samples=750, ftype=".bin"):

    # generate or load model
    base_model_folder = "data/rdf/trainedmodels/"
    seg_model_path = os.path.join(base_model_folder,
                                  "{:s}_M{:d}_rad{:0.2f}".format("mixed_all_participants", M, radius))
    rdf_hs = load_hs_model("RDF Model", offset_gen, feature_gen, M, radius, n_samples, refresh=False,
                           segmentation_model_path=seg_model_path)

    return HandDetector(rdf_hs)

def visualize(dmap, hd):

    lh, rh = hd.detect_hands(dmap)

    ### Visualize OG
    print("Vis OG")
    plt.imshow(lh.get_original(), vmin=0, vmax=850)
    plt.show()

    ### Visualize the LH and RH Hand Masks
    print("Vis Mask")
    mask_rgb = np.zeros((lh.mask().shape[0], lh.mask().shape[1], 3), dtype=float)

    mask_rgb[:, :, 0] = lh.mask()
    mask_rgb[:, :, 1] = rh.mask()

    mask_rgb[mask_rgb > 0] = 1

    plt.imshow(mask_rgb)

    # Add the lh bounding box
    box = lh.hand_box()
    x, y  = box[0], box[1]
    width, height = box[2], box[3]
    rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

    # Add the re bounding box
    box = rh.hand_box()
    x, y  = box[0], box[1]
    width, height = box[2], box[3]
    rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='b',facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

    plt.show()

    ### Visualize each hand
    print("Vis Hands")
    plt.imshow(lh.get_hand_img(), vmin=0, vmax=lh.get_hand_img().max())
    plt.show()

    plt.imshow(rh.get_hand_img(), vmin=0, vmax=rh.get_hand_img().max())
    plt.show()

    ### Visualize Features
    print("Vis Features")
    x_lh, lh_vis = extract_features(lh.get_hand_img(), "test", visualise=True)
    x_rh, rh_vis = extract_features(rh.get_hand_img(), "test", visualise=True)

    lh_vis_sq = np.ma.log(lh_vis)
    rh_vis_sq = np.ma.log(rh_vis)

    lh_vis_sq = lh_vis_sq.filled(-50)
    rh_vis_sq = rh_vis_sq.filled(-50)

    print("LH MIN:", lh_vis[lh_vis>0].min(), lh_vis.max())
    print("LH LOG:", lh_vis_sq.min(), lh_vis_sq.max())
    print("RH REG:", rh_vis[rh_vis>0].min(), rh_vis.max())
    print("RH LOG:", rh_vis_sq.min(), rh_vis_sq.max())

    plt.figure()
    plt.subplot(121)
    plt.imshow(lh_vis, vmin=lh_vis.min(), vmax=lh_vis.max())
    plt.subplot(122)
    plt.imshow(lh_vis_sq, vmin=lh_vis_sq.min(), vmax=lh_vis_sq.max())
    plt.show()

    plt.subplot(121)
    plt.imshow(rh_vis, vmin=rh_vis.min(), vmax=rh_vis.max())
    plt.subplot(122)
    plt.imshow(rh_vis_sq, vmin=rh_vis_sq.min(), vmax=rh_vis_sq.max())
    plt.show()


def main():

    offset_gen = dcf.generate_feature_offsets
    feature_gen = dcf.calc_features

    # RDF Parameters
    M = 5
    radius = .15

    # Posture Detection Parameters
    feature_type = "test"

    dmap = load_depthmap("data/posture/p4/p4e/000200.bin")

    hd = get_hd(offset_gen, feature_gen, M, radius, n_samples=750, ftype=".bin")
    visualize(dmap, hd)

if __name__ == '__main__':
    main()
