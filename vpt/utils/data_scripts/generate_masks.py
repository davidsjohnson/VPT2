from vpt.streams.file_stream import FileStream
from vpt.hand_detection.sub_segmentation_model import *
from vpt.hand_detection.hand_detector import *
from vpt.hand_detection.hand_generator import *
from vpt.hand_detection.depth_context_features import *
from vpt.common import *
import vpt.settings

from skimage.transform import rescale

def rotate_hand(hand_img, angle=20.0, scale=.5):

    center = (hand_img.shape[0]/2, hand_img.shape[1]/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    return cv2.warpAffine(hand_img, rot_mat, hand_img.shape)

def superimpose_hands(lh, rh, bg_dmap, hand_sensor, bg_sensor, y_shift_amount=30):

    bg_dmap = bg_dmap.copy()
    lh_mask = lh.get_mask()
    rh_mask = rh.get_mask()

    lh_dmap = cv2.bitwise_and(lh.get_original(), lh.get_original(), mask=lh_mask)
    rh_dmap = cv2.bitwise_and(rh.get_original(), rh.get_original(), mask=rh_mask)

    # #project hand data to same space as sensor data
    # tmp_sensor = vpt.settings.sensor
    # vpt.settings.sensor = hand_sensor
    # sample_mask = np.ones_like(lh_dmap, dtype=bool)
    #
    # lh_points = pixels2points(lh_dmap, sample_mask)
    # lh_points = np.expand_dims(lh_points, 1)
    #
    # rh_points = pixels2points(rh_dmap, sample_mask)
    # rh_points = np.expand_dims(rh_points, 1)
    #
    # vpt.settings.sensor = bg_sensor
    # lh_pixels, lh_depth = points2pixels(lh_points, True)
    # lh_pixels = np.squeeze(lh_pixels)
    # lh_depth = np.squeeze(lh_depth)
    #
    # rh_pixels, rh_depth = points2pixels(rh_points, True)
    # rh_pixels = np.squeeze(rh_pixels)
    # rh_depth = np.squeeze(rh_depth)
    #
    # lh_tmp_dmap = np.zeros_like(lh_dmap)
    # lh_tmp_dmap[(lh_pixels[:, 1], lh_pixels[:, 0])] = lh_depth
    # lh_dmap = lh_tmp_dmap
    #
    # rh_tmp_dmap = np.zeros_like(rh_dmap)
    # rh_tmp_dmap[(rh_pixels[:, 1], rh_pixels[:, 0])] = rh_depth
    # rh_dmap = rh_tmp_dmap
    #
    # vpt.settings.sensor = tmp_sensor

    # find average depth of the tip of hand to place at correct height when superimposing
    lh_handtip = lh.get_hand_img().copy()[-10:, :].astype(float)
    rh_handtip = rh.get_hand_img().copy()[-10:, :].astype(float)
    lh_handtip[lh_handtip == 0] = np.nan
    rh_handtip[rh_handtip == 0] = np.nan

    # 30 found empirically
    lh_avg = np.nanmean(lh_handtip) - 30
    rh_avg = np.nanmean(rh_handtip) - 30
    bg_min_val = bg_dmap[bg_dmap>0].min()

    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(lh_dmap)
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(lh.get_hand_img().copy()[-10:, :])
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(bg_dmap)
    # plt.colorbar()

    lh_dmap = lh_dmap[:-y_shift_amount, :]
    rh_dmap = rh_dmap[:-y_shift_amount, :]

    bg_dmap[y_shift_amount:, :][lh_dmap > 0] = lh_dmap[lh_dmap > 0] +  bg_min_val - lh_avg
    bg_dmap[y_shift_amount:, :][rh_dmap > 0] = rh_dmap[rh_dmap > 0] +  bg_min_val - rh_avg

    # plt.subplot(224)
    # plt.imshow(bg_dmap)
    # plt.colorbar()
    # plt.show()

    return bg_dmap


def transform_hands(lh, rh, bg_dmap, y_shift_amount, transform_func, *args, **kwargs):

    lh_box = lh.hand_box(None)
    rh_box = rh.hand_box(None)

    lh_dmap = lh.get_hand_img().copy()
    rh_dmap = rh.get_hand_img().copy()

    lh_mask = lh.get_mask()
    rh_mask = rh.get_mask()

    #### transform hand depth maps
    lh_tmp = np.zeros_like(lh_dmap)
    rh_tmp = np.zeros_like(rh_dmap)

    lh_dmap = transform_func(lh_dmap, *args, **kwargs)
    rh_dmap = transform_func(rh_dmap, *args, **kwargs)

    lh_y_diff = 0
    lh_x_diff = 0
    rh_y_diff = 0
    rh_x_diff = 0

    if lh_dmap.shape > lh_tmp.shape and rh_dmap.shape > rh_tmp.shape:

        lh_y_diff = lh_dmap.shape[0] - lh_tmp.shape[0]
        lh_x_diff = lh_dmap.shape[1] - lh_tmp.shape[1]
        rh_y_diff = rh_dmap.shape[0] - rh_tmp.shape[0]
        rh_x_diff = rh_dmap.shape[1] - rh_tmp.shape[1]

        lh_tmp = np.zeros_like(lh_dmap)
        rh_tmp = np.zeros_like(rh_dmap)

    lh_x1 = lh_box[0]
    lh_y1 = lh_box[1]
    lh_x2 = lh_x1 + lh_box[2] + lh_x_diff
    lh_y2 = lh_y1 + lh_box[3] + lh_y_diff

    rh_x1 = rh_box[0]
    rh_y1 = rh_box[1]
    rh_x2 = rh_x1 + rh_box[2] + rh_x_diff
    rh_y2 = rh_y1 + rh_box[3] + rh_y_diff

    lh_tmp[-lh_dmap.shape[0] : , -lh_dmap.shape[1] : ] = lh_dmap
    rh_tmp[-rh_dmap.shape[0] : , -rh_dmap.shape[1] : ] = rh_dmap

    # create new depthmap with transformed hands (resulting image will have holes where previous hand was)
    dmap_new = lh.get_original().copy()  # doesn't matter which hand since lh and rh from same original img

    dmap_new[lh_mask > 0] = 0
    dmap_new[rh_mask > 0] = 0

    dmap_new[lh_y1:lh_y2, lh_x1:lh_x2][lh_tmp>0] = lh_tmp[lh_tmp>0]
    dmap_new[rh_y1:rh_y2, rh_x1:rh_x2][rh_tmp>0] = rh_tmp[rh_tmp>0]

    #### transform hand mask
    lh_tmp_mask = np.zeros_like(lh_mask[lh_y1:lh_y2-lh_y_diff, lh_x1:lh_x2-lh_x_diff])
    rh_tmp_mask = np.zeros_like(rh_mask[rh_y1:rh_y2-rh_y_diff, rh_x1:rh_x2-rh_x_diff])

    lh_mask_res = transform_func(lh_mask[lh_y1:lh_y2-lh_y_diff, lh_x1:lh_x2-lh_x_diff], *args, **kwargs)
    rh_mask_res = transform_func(rh_mask[rh_y1:rh_y2-rh_y_diff, rh_x1:rh_x2-rh_x_diff], *args, **kwargs)

    if lh_mask_res.shape > lh_tmp_mask.shape and rh_mask_res.shape > rh_tmp_mask.shape:
        lh_tmp_mask = np.zeros_like(lh_mask[lh_y1:lh_y2, lh_x1:lh_x2])
        rh_tmp_mask = np.zeros_like(rh_mask[rh_y1:rh_y2, rh_x1:rh_x2])

    lh_tmp_mask[-lh_mask_res.shape[0] : , -lh_mask_res.shape[1] : ] = lh_mask_res
    rh_tmp_mask[-rh_mask_res.shape[0] : , -rh_mask_res.shape[1] : ] = rh_mask_res

    lh_mask_new = np.zeros_like(lh_mask)
    rh_mask_new = np.zeros_like(rh_mask)

    lh_mask_new[lh_y1:lh_y2, lh_x1:lh_x2] = lh_tmp_mask
    rh_mask_new[rh_y1:rh_y2, rh_x1:rh_x2] = rh_tmp_mask

    # calculate new handbox
    lh_box_new = [lh_x2-lh_mask_res.shape[1], lh_y2-lh_mask_res.shape[0], lh_mask_res.shape[1], lh_mask_res.shape[0]]
    rh_box_new = [rh_x2-rh_mask_res.shape[1], rh_y2-rh_mask_res.shape[0], rh_mask_res.shape[1], rh_mask_res.shape[0]]

    # create new hand instances
    lh_new = Hand(lh_mask_new, dmap_new, lh_box_new, fpath=lh.get_fpath() + "transformed", label=lh.label())
    rh_new = Hand(rh_mask_new, dmap_new, rh_box_new, fpath=rh.get_fpath() + "transformed", label=rh.label())

    # create new "original image" with transformed hands
    new_dmap = superimpose_hands(lh_new, rh_new, bg_dmap, 'kinect', 'realsense', y_shift_amount=y_shift_amount)

    new_lh_mask = np.zeros_like(lh_new.mask())
    new_rh_mask = np.zeros_like(rh_new.mask())

    new_lh_mask[y_shift_amount:, :] = lh_new.mask()[:-y_shift_amount, :]
    new_rh_mask[y_shift_amount:, :] = rh_new.mask()[:-y_shift_amount, :]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new_lh_mask = cv2.morphologyEx(new_lh_mask, cv2.MORPH_OPEN, kernel)  # perform Morphological opening to remove noise
    new_rh_mask = cv2.morphologyEx(new_rh_mask, cv2.MORPH_OPEN, kernel)

    lh_new.dmap(new_dmap)
    lh_new.mask(new_lh_mask)
    lh_box = lh_new.hand_box()
    lh_box[1]+=y_shift_amount
    lh_new.hand_box(lh_box)

    rh_new.dmap(new_dmap)
    rh_new.mask(new_rh_mask)
    rh_box = rh_new.hand_box()
    rh_box[1]+=y_shift_amount
    rh_new.hand_box(rh_box)

    return lh_new, rh_new

def masks2disk(lh, rh, hand_num, folder, tag):

    # add masks to full image
    mask = np.zeros((lh.mask().shape[0], lh.mask().shape[1], 3))
    mask[:, :, 0] = lh.mask()
    mask[:, :, 2] = rh.mask()

    #save to disk as jpg
    file_name = "{:06d}".format(hand_num)

    participant = vpt.settings.participant

    mask_folder = os.path.join(folder, participant, tag, "mask")
    orig_folder = os.path.join(folder, participant, tag, "orig")

    mask_path = os.path.join(mask_folder, file_name+".bmp")
    orig_path = os.path.join(orig_folder, file_name+".npy")

    if not os.path.exists(mask_folder): os.makedirs(mask_folder)
    if not os.path.exists(orig_folder): os.makedirs(orig_folder)

    cv2.imwrite(mask_path, mask)
    np.save(orig_path, lh.dmap())



def get_files(folder):
    import re

    var_threshs = np.linspace(.7, 1, 5)
    errors = ["error0", "error1", "error2"]

    file_nums = {}

    for v in var_threshs:
        for e in errors:
            regex = '[a-zA-Z0-9_/]*/{error}/scaled0\.7{thresh}/mask/(\d{{6}}).jpg'.format(error=e, thresh=str(v).replace(".", "\."))

            for root, dirs, files in os.walk(folder):
                for name in files:

                    path = os.path.join(root, name)
                    match = re.match(regex, path)

                    if match:
                        key = (e, v)
                        if key not in file_nums:
                            file_nums[key] = []

                        file_nums[key].append(int(match.groups()[0]))

    return file_nums


def depth_scaling(dmap, *args, **kwargs):

    scaled = rescale(dmap, *args, **kwargs)
    return scaled*kwargs.get('scale')


def main(data_folder, background_folder, ftype, annotation_file, tag, transform_func, y_shift_amount=30, *args, **kwargs):

    # var_threshs = np.linspace(.7, 1, 5)
    var_threshs = [.925, 1.0]
    wait_char = None

    file_nums = get_files('data/rdf/generated/p01')

    for var_thresh in var_threshs:

        print("Starting Threshold: {:.4}".format(var_thresh))

        annotations = load_annotations(annotation_file)
        fs = FileStream(data_folder, ftype=ftype, annotations=annotations, ignore=True, normalize=False)

        hs = SubSegmentationModel("{}_testmodel".format(vpt.settings.participant))
        hs.initialize(background_folder, varThreshold=var_thresh, history=75)
        hd = HandDetector(hs)
        hg = HandGenerator(fs, hd, annotations)

        h_gen = hg.hand_generator(debug=False)

        bg_dmap = load_depthmap("data/backgrounds/p4/p4bs/000061.bin", normalize=False)

        error_num = tag.split("/")[0]

        for i, (lh, rh) in enumerate(h_gen):

            if i in file_nums[(error_num, var_thresh)]:

                lh_new, rh_new = transform_hands(lh, rh, bg_dmap, y_shift_amount, transform_func, *args, **kwargs)

                #TODO: Add imshow for depthmaps
                lh_img = (ip.normalize(lh_new.get_hand_img())*255).astype('uint8')
                rh_img = (ip.normalize(rh_new.get_hand_img())*255).astype('uint8')
                dmap_img = (ip.normalize(rh.dmap()) * 255).astype('uint8')

                masks2disk(lh_new, rh_new, i, folder="data/rdf/generated",
                           tag=tag + str(kwargs.get('scale', "")) + str(var_thresh))

                # cv2.imshow("LH", lh_img)
                # cv2.imshow("RH", rh_img)
                # cv2.imshow("LH Mask", lh_new.mask())
                # cv2.imshow("RH Mask", rh_new.mask())
                # cv2.imshow("OG DMap", dmap_img)

            #     wait_char = cv2.waitKey(0)
            #     if wait_char == ord('q') or wait_char == ord('n'):
            #         break
            #     elif wait_char == ord('s'):
            #         print("Saving Mask...")
            #         masks2disk(lh_new, rh_new, i, folder="data/rdf/generated", tag=tag + str(kwargs.get('scale', "")) + str(var_thresh))
            #
            # if wait_char == ord('q'): break


if __name__ == "__main__":

    import argparse

    scale_factors = [1.3]

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-folder", help="Root Folder containing data to process", required=True)
    parser.add_argument("-b", "--background-folder", help="Folder containing background for background subtraction", required=True)
    parser.add_argument("-p", "--participant", help="Participant Identifier", required=True)
    parser.add_argument("-s", "--sensor-type", help="Sensor type used to capture the data", required=True)
    parser.add_argument("-t", "--data-type", help="Data stream file type", default="bin")
    parser.add_argument("-a", "--annotations", help="Annotation file for provided data", required=True)

    args = parser.parse_args()

    vpt.settings.participant = args.participant
    vpt.settings.sensor = args.sensor_type

    # TODO::START HERE AND CHANGE THIS
    # TODO:::::New Scale Value
    # TODO:::::use existing selections to help with new selections (instead of manual mask selection)

    for scale_factor in scale_factors:
        print ("Scaling By:", scale_factor)
        main(args.data_folder, args.background_folder, args.data_type, args.annotations, "error2/scaled", transform_func=depth_scaling, scale=scale_factor, preserve_range=True, order=0)