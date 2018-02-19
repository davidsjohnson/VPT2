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
    np.random.seed(0)
    return np.random.randint(-radius, radius, size=(2, n_offsets, 2))


def calc_features(dmap, offsets, sample_mask=None):

    start_time = time.time()
    dmap = dmap.astype(int)

    if sample_mask is None:
        sample_mask = np.ones_like(dmap, dtype=bool)

    dmap[dmap==0] = 10000

    offsets_u = offsets[0]
    offsets_v = offsets[1]

    dmap_ = dmap[sample_mask].ravel()
    offsets_u_ = np.repeat(offsets_u[np.newaxis, :], len(dmap_), axis=0)
    offsets_v_ = np.repeat(offsets_v[np.newaxis, :], len(dmap_), axis=0)
    dmap_ = np.repeat(dmap_[:, np.newaxis], offsets_u.shape[0], axis=1)
    dmap_ = np.repeat(dmap_[:, :, np.newaxis], offsets_u.shape[1], axis=2)

    offsets_u_ = offsets_u_ / dmap_
    offsets_v_ = offsets_v_ / dmap_

    indexes = np.indices(dmap.shape)
    indexes = np.moveaxis(indexes, 0, 2)
    indexes = np.reshape(indexes, (indexes.shape[0] * indexes.shape[1], indexes.shape[2]))
    indexes = np.repeat(indexes[:, np.newaxis, :], offsets_u.shape[0], axis=1)

    indexes_u = indexes[sample_mask.ravel()] + offsets_u_
    indexes_v = indexes[sample_mask.ravel()] + offsets_v_

    indexes_u = indexes_u.astype(int)
    indexes_v = indexes_v.astype(int)

    # Verify calculated offsets are inbounds of dmap
    indexes_u[:, :, 0][indexes_u[:, :, 0] >= dmap.shape[0]] = -1
    indexes_u[:, :, 1][indexes_u[:, :, 1] >= dmap.shape[1]] = -1
    indexes_u[indexes_u < 0] = -1
    indexes_v[:, :, 0][indexes_v[:, :, 0] >= dmap.shape[0]] = -1
    indexes_v[:, :, 1][indexes_v[:, :, 1] >= dmap.shape[1]] = -1
    indexes_v[indexes_v < 0] = -1

    d_u = np.ones((offsets_u_.shape[0], offsets_u_.shape[1])) * MAX_FEATURE_VAL
    d_v = np.ones((offsets_v_.shape[0], offsets_v_.shape[1])) * MAX_FEATURE_VAL

    i_u = np.all(indexes_u != -1, axis=2)
    i_v = np.all(indexes_v != -1, axis=2)

    d_u[i_u] = dmap[indexes_u[i_u][:, 0], indexes_u[i_u][:, 1]]
    d_v[i_v] = dmap[indexes_v[i_v][:, 0], indexes_v[i_v][:, 1]]

    features = d_u - d_v

    # print("Total Features Time:", time.time() - start_time)
    return features, None


if __name__ == '__main__':

    s.sensor = "realsense"
    n_samples = 500
    n_features = 50
    offsets = generate_feature_offsets(n_features, 100000)

    hands = np.load("data/rdf/training/p6/000072.npz")
    dmap = hands["dmap"]
    mask = hands['mask']

    pixels = np.where(mask[:,:,0] > 0)
    pixels = np.hstack((np.expand_dims(pixels[1], 1), np.expand_dims(pixels[0], 1)))  # combine results to x, y pairs

    sample_idxs = np.arange(pixels.shape[0])
    np.random.seed(0)
    np.random.shuffle(sample_idxs)
    sample_idxs = sample_idxs[:n_samples]

    sample_pixels = pixels[sample_idxs]
    sample_mask = np.zeros_like(dmap, dtype=bool)

    sample_mask[sample_pixels[:, 1:], sample_pixels[:, :1]] = True

    features = calc_features(dmap, offsets, sample_mask)

    print("Avg", np.average(features))
    print("Max", features.max())
    print("Min", features.min())
    print(features.shape)
    print(np.average(features, axis=1).shape)
    print(sample_mask[sample_mask].ravel().shape)

    avg = np.zeros_like(dmap, dtype="int")
    avg = avg.ravel()
    avg[sample_mask.ravel()] = np.average(features, axis=1)
    avg = np.reshape(avg, dmap.shape)

    print("Avg Max", avg.max())

    # print(features.shape)
    # features = np.reshape(features, (dmap.shape[0], dmap.shape[1], n_features))
    # avg = np.average(features, axis=2)
    # print(avg.shape)

    plt.figure()
    plt.subplot(121)
    plt.imshow(sample_mask)
    # plt.colorbar()
    plt.subplot(122)
    plt.imshow(ip.normalize2(avg))
    # plt.colorbar()
    plt.show()