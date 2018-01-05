# David Johnson
#   data_augmentation.py
#
#   scripts and functions used to generate an augmented RDF data set from a set of original data for hand segmentation

from skimage.transform import rescale

from vpt.common import *
from vpt.streams.mask_stream import MaskStream
from vpt.hand_detection.hand import Hand

import vpt.settings as s


# MASK_FOLDERS = {"p1" : "data/rdf/p1/seq_masks", "p2" : "data/rdf/p2/seq_masks", "p3" : "data/rdf/p3/seq_masks",
#                 "p4" : "data/rdf/p4/seq_masks", "p6" : "data/rdf/p6/seq_masks"}

MASK_FOLDERS = {"p6" : "data/rdf/p6/seq_masks"}


def get_bounding_box(mask):

    if cv2.__version__ < '3.0':
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        img, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_rect = None

    for c in contours:

        # find bounding box and area
        poly = cv2.approxPolyDP(c, epsilon=3, closed=True)
        rect = cv2.boundingRect(poly)
        area = rect[2] * rect[3]

        if area > max_area:
            max_area = area
            max_rect = rect

    return max_rect


def scale(img, *args, **kwargs):

    zfactor = kwargs.get("zfactor", 1)
    del kwargs["zfactor"]
    scaled = rescale(img, *args, **kwargs)
    scaled = scaled * zfactor
    return scaled


def transform_hands(lh, rh, dmap_bkgd, transform_func, *args, **kwargs):
    '''Applies a transformation function to both the right and left hands and superimposes the hands on a background depth map'''

    #### transform hand images
    dmap_lh_trans = transform_func(lh.get_hand_img(), *args, **kwargs)
    dmap_rh_trans = transform_func(rh.get_hand_img(), *args, **kwargs)



    ## place hands on background image; anchored at bottom left of original hand box location
    # get new X and Y coords such that transformed dmap is anchored to bottom left
    x1_lh = lh.hand_box()[0] + lh.hand_box()[2] - dmap_lh_trans.shape[1]
    y1_lh = lh.hand_box()[1] + lh.hand_box()[3] - dmap_lh_trans.shape[0]
    x1_rh = rh.hand_box()[0] + rh.hand_box()[2] - dmap_rh_trans.shape[1]
    y1_rh = rh.hand_box()[1] + rh.hand_box()[3] - dmap_rh_trans.shape[0]

    # shift hands horizontally if too close (shift values determined empirically)
    if x1_rh - (x1_lh + dmap_lh_trans.shape[1]) < 10:
        x1_lh -= 50
        x1_rh += 50

    # if x1_lh is less than zero shift right to zero
    if x1_lh < 0:
        x1_lh = 0

    #if y1_lh is less than zero truncate hand box
    if y1_lh < 0:
        dmap_lh_trans = dmap_lh_trans[abs(y1_lh) : , : ]
        y1_lh = 10

    # if x2_rh is greater than background width than shift left to fit
    if x1_rh + dmap_rh_trans.shape[1] > dmap_bkgd.shape[1]:
        x1_rh -= (x1_rh + dmap_rh_trans.shape[1] - dmap_bkgd.shape[1])

    #if y1_rh is less than zero truncate hand box
    if y1_rh < 0:
        dmap_rh_trans = dmap_rh_trans[abs(y1_rh) : , : ]
        y1_rh = 10

    # if y1 + height is greater than dmap height shift up
    if y1_lh + dmap_lh_trans.shape[0] > dmap_bkgd.shape[0]:
        y1_lh -= (y1_lh + dmap_lh_trans.shape[0] - dmap_bkgd.shape[0])

    if y1_rh + dmap_rh_trans.shape[0] > dmap_bkgd.shape[0]:
        y1_rh -= (y1_rh + dmap_rh_trans.shape[0] - dmap_bkgd.shape[0])

    #### add the new transformed hand to the background
    # get the hand box location from the background dmap
    dmap_new = dmap_bkgd.copy()
    lh_tmp_hand = dmap_new[y1_lh : y1_lh + dmap_lh_trans.shape[0], x1_lh : x1_lh + dmap_lh_trans.shape[1]]
    rh_tmp_hand = dmap_new[y1_rh : y1_rh + dmap_rh_trans.shape[0], x1_rh : x1_rh + dmap_rh_trans.shape[1]]

    # print(y1_lh , y1_lh + dmap_lh_trans.shape[0], x1_lh , x1_lh + dmap_lh_trans.shape[1])

    # place transformed hands at same depth as piano
    diff_lh = lh_tmp_hand[lh_tmp_hand > 0].min() - dmap_lh_trans[ -20 :, :].max()     # use the max value of the finger tips instead of global max (ie -20)
    diff_rh = rh_tmp_hand[rh_tmp_hand > 0].min() - dmap_rh_trans[ -20 :, :].max()     # use the max value of the finger tips instead of global max (ie -20)
    dmap_lh_trans[dmap_lh_trans > 0] += diff_lh
    dmap_rh_trans[dmap_rh_trans > 0] += diff_rh
    lh_tmp_hand[dmap_lh_trans > 0] = dmap_lh_trans[dmap_lh_trans > 0]
    rh_tmp_hand[dmap_rh_trans > 0] = dmap_rh_trans[dmap_rh_trans > 0]

    #### transform hand masks
    mask_new = np.zeros((dmap_bkgd.shape[0], dmap_bkgd.shape[1], 3), dtype="uint8")
    lh_tmp_mask = mask_new[:, :, s.LH][y1_lh : y1_lh + dmap_lh_trans.shape[0], x1_lh : x1_lh + dmap_lh_trans.shape[1]]
    rh_tmp_mask = mask_new[:, :, s.RH][y1_rh : y1_rh + dmap_rh_trans.shape[0], x1_rh : x1_rh + dmap_rh_trans.shape[1]]
    lh_tmp_mask[dmap_lh_trans > 0] = 255
    rh_tmp_mask[dmap_rh_trans > 0] = 255

    return dmap_new.astype("uint16"), mask_new


def main():

    base_folder = "data/rdf/training"
    s.sensor = "realsense"

    scale_factors = [.75, .875, 1.125, 1.25]

    for participant, folder in MASK_FOLDERS.items():


        s.participant = participant
        fs = MaskStream(MASK_FOLDERS[s.participant])
        dmap_bkgd = load_depthmap("data/backgrounds/{}/{}bs/000240.bin".format(s.participant, s.participant), "bin", False)

        i = 0
        mgen = fs.img_generator()
        for mask, dmap, fpath in mgen:

            # save original to training data folder
            fname = "{:06d}.npz".format(i)
            fpath = os.path.join(base_folder, s.participant, fname)
            np.savez_compressed(fpath, dmap=dmap, mask=mask)
            i += 1

            ##### Create augmented data
            # Create Hand Object for Transform Function
            lh_box = get_bounding_box(mask[:, :, s.LH])
            rh_box = get_bounding_box(mask[:, :, s.RH])
            lh = Hand(mask[:, :, s.LH], dmap, lh_box, fpath)
            rh = Hand(mask[:, :, s.RH], dmap, rh_box, fpath)

            # apply transformations to each image
            for x in scale_factors:
                for y in scale_factors:
                    for z in scale_factors:
                        # try:
                            dmap_new, mask_new = transform_hands(lh, rh, dmap_bkgd, transform_func=scale, scale=(y, x), zfactor=z, preserve_range=True, order=0)

                            # save transformation
                            fname = "{:06d}.npz".format(i)
                            fpath = os.path.join(base_folder, s.participant, fname)
                            np.savez_compressed(fpath, dmap=dmap_new, mask=mask_new)
                            i += 1
                        # except IndexError as e:
                        #     print("Error Creating transformation:", e)

if __name__ == '__main__':
    main()