# David Johnson
#   depth_image_features.py
#   Functions to generate features from depth pixels. Features are caclulated using the a variation
#   on the technique from "Real-Time Human Pose Recognition in Parts from Single Depth Images"

from vpt.common import *
import vpt.settings as s

import time

MAX_FEATURE_VAL = 10000

def generate_feature_offsets(n_offsets, radius):
    '''
        Generates random offset points in depth map space
        used to calculate depth image features from a given pixel

        n_offsets: Number of offsets to generate
        radius: size of offset coverage in pixel meters ??
    '''

    return np.random.randint(-radius, radius, size=(n_offsets, 2, 2))


def calc_feature(o, depth, dmap, pixel):

    x_u = pixel[1] + o[0][1] // depth
    y_u = pixel[0] + o[0][0] // depth
    x_v = pixel[1] + o[1][1] // depth
    y_v = pixel[0] + o[1][0] // depth

    # check if offset is within depth map bounds set to a large value if not
    if x_u < 0 or x_u >= dmap.shape[1] or y_u < 0 or y_u >= dmap.shape[0]:
        d_u = MAX_FEATURE_VAL
    else:
        d_u = dmap[y_u, x_u] if dmap[y_u, x_u] != 0 else MAX_FEATURE_VAL

    if x_v < 0 or x_v >= dmap.shape[1] or y_v < 0 or y_v >= dmap.shape[0]:
        d_v = MAX_FEATURE_VAL
    else:
        d_v = dmap[y_v, x_v] if dmap[y_v, x_v] != 0 else MAX_FEATURE_VAL

    return d_u - d_v


def calc_pixel_features(pixel, dmap, offsets):

        # start_time = time.time()
        depth = dmap[tuple(pixel)] if dmap[tuple(pixel)] != 0 else MAX_FEATURE_VAL

        features = list(map(lambda o: calc_feature(o, depth, dmap, pixel), offsets))
        # features = [calc_feature(o, depth, dmap, pixel) for o in offsets]

        # print("Total Time for Pixel:", time.time()-start_time)
        return np.array(features)


def calc_features(dmap, offsets, sample_mask=None):

    dmap = dmap.astype(int)

    start_time = time.time()
    if sample_mask is None:
        sample_mask = np.ones_like(dmap, dtype=bool)

    sample_mask = sample_mask.ravel()

    # get pixel values
    pixels = np.indices(dmap.shape, dtype=int)
    pixels = np.moveaxis(pixels, 0, 2)
    pixels = pixels.reshape((pixels.shape[0]*pixels.shape[1], 2))[sample_mask]

    # calculate depth feature for all pixels
    features = list(map(lambda p: calc_pixel_features(p, dmap, offsets), pixels))
    # features = [calc_pixel_features(p, dmap, offsets) for p in pixels]
    # print("Total Time for Depth Map:", time.time() - start_time)

    return np.array(features)


if __name__ == '__main__':

    s.sensor = "realsense"
    dmap = load_depthmap("data/posture/p1/p1b/000251.bin")
    offsets = generate_feature_offsets(100, 50000)

    features = calc_features(dmap, offsets)
