from vpt.common import *

import math

from skimage.transform import resize
from skimage.feature import hog as skhog
from skimage.util import pad

from vpt.hand_detection.hand_generator import *
import vpt.settings as s

def hog(img, visualise=False, pixels_per_cell=(6,6), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(180,180), pad_img=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if not pad_img:
            img = resize(img, img_size)
        else:
            # the rare case that a hand is bigger than new size
            if img.shape[0] > img_size[0] and img.shape[1] > img_size[1]:
                img = resize(img, img_size)

            y_dif = img_size[0] - img.shape[0]
            x_dif = img_size[1] - img.shape[1]
            img = pad(img, ((y_dif, 0), (x_dif, 0)), mode='constant')

        return skhog(img, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm=block_norm, visualise=visualise)


def sliced_hog(img, n_slices=20, visualise=False, cells_per_block=(1,1), block_norm="L1-sqrt"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = resize(img, (120, 90))
        cell_size = (img.shape[1], img.shape[0] / float(n_slices))
        # hog = skhog(img, orientations=9, pixels_per_cell=cell_size, cells_per_block=(1,4), block_norm="L1-sqrt", visualise=visualise)
        hog = skhog(img, orientations=9, pixels_per_cell=cell_size, cells_per_block=cells_per_block, block_norm=block_norm, visualise=visualise)  #no-block


    if visualise:
        return hog[0], hog[1]
    else:
        return hog

def cae(X, encoder):
    return encoder.predict(X)


def shonv(img, num_bins = 10, num_slices = 10):

    img = ip.normalize(img)

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    azi = np.arctan2(gy, gx)
    azi[azi < 0] += 2 * np.pi
    zen = np.arctan(np.sqrt(gx * gx + gy * gy))

    slice_size = img.shape[0] // num_slices

    hists = []
    for i in range(0, num_slices):
        azi_slice = azi[i * slice_size:i * slice_size + slice_size - 1, :]
        zen_slice = zen[i * slice_size:i * slice_size + slice_size - 1, :]

        H, xedges, yedges = np.histogram2d(azi_slice.ravel(), zen_slice.ravel(), bins=num_bins)

        hists.append(np.hstack(H))

    return np.hstack(hists)



## Histograms of Oriented Normal Vectors

def sliding_window(image1, step_size, window_size):

    for y in range(0, image1.shape[0], step_size[0]):
        for x in range(0, image1.shape[1], step_size[1]):
            # yield the current window
            yield image1[y:y + window_size[1], x:x + window_size[0]]


def honv(img, pixels_per_cell=(6, 6), cells_per_block=(1,1), num_bins=9, img_size=(180, 180), pad_img=False, block_norm=None):

    img = img.astype(float)

    if not pad_img:
        img = resize(img, img_size, mode="constant")
    else:
        # the rare case that a hand is bigger than new size
        if img.shape[0] > img_size[0] and img.shape[1] > img_size[1]:
            img = resize(img, img_size, mode="constant")

        y_dif = img_size[0] - img.shape[0]
        x_dif = img_size[1] - img.shape[1]
        img = pad(img, ((y_dif, 0), (x_dif, 0)), mode='constant')

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    azi = np.arctan2(gy, gx)
    azi[azi < 0] += 2 * np.pi
    zen = np.arctan(np.sqrt(gx * gx + gy * gy))

    hists = []
    for azi_cell, zen_cell in zip(sliding_window(azi, step_size=pixels_per_cell, window_size=pixels_per_cell), sliding_window(zen, step_size=pixels_per_cell, window_size=pixels_per_cell)):

        try:
            H, xedges, yedges = np.histogram2d(azi_cell.ravel(), zen_cell.ravel(), range=[[0, 6.3], [0, 1.6]], bins=num_bins) #range values found empirically
            hists.append(np.hstack(H))
        except Exception as e:
            print(e)
            print(azi_cell)
            print(zen_cell)
            print()

    if block_norm != None:
        normed_hists = []
        sigma = .01

        ceildiv = lambda a, b: -(-a // b)  # ceiling divide to round up for histogram reshaping
        hists_tmp = np.reshape(hists, (ceildiv(img_size[0], pixels_per_cell[0]), ceildiv(img_size[1], pixels_per_cell[1]), -1))
        step_size = (2,2)
        if cells_per_block == (1,1) or cells_per_block == (2,2):
            step_size = (1,1)
        for tmp in sliding_window(hists_tmp, step_size=step_size, window_size=cells_per_block):

            if block_norm == "L1-sqrt":
                v = tmp.ravel()
                v = np.sqrt(v / (np.linalg.norm(v, ord=1) + sigma))
                normed_hists.append(v)
            else:
                raise ValueError("Invalid Norm Type")

        hists = normed_hists

    return np.hstack(hists).astype("float32")

# def sliced_hog(img, n_slices=20):
#     ''' Horizontally sliced histograms of Oriented Gradients '''
#
#     img = (ip.normalize(img)*255).astype('uint8')
#     # img = img.astype(float)
#
#     num_bins = 16
#
#     slice_size = img.shape[0] / n_slices
#
#     hists = []
#
#     # kx, ky = cv2.getDerivKernels(1,1,1)
#     # kx = kx[::-1].transpose()
#     # gx = cv2.filter2D(img, -1, kx, )
#     # gy = cv2.filter2D(img,-1, ky, cv2.CV_32F)
#
#
#     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     mag, ang = cv2.cartToPolar(gx, gy)
#
#     bins = np.int32(num_bins*ang/(2*np.pi))
#
#     for i in range(0, n_slices):
#
#         bin_slice = bins[i*slice_size:i*slice_size+slice_size-1, :]
#         mag_slice = mag[i*slice_size:i*slice_size+slice_size-1, :]
#
#         hist = np.bincount(bin_slice.ravel(), mag_slice.ravel(), num_bins)
#         hists.append(hist)
#
#         # hists.append(hog(img[i*slice_size:i*slice_size+slice_size-1, :]))
#
#     return np.hstack(hists)


def generate_data_set(hands, xtype="shog", training=True):

    X = []
    y = []

    for i, h in enumerate(hands):

        if xtype == "shog":
            x = sliced_hog(h.get_hand_img())
        elif xtype == "hog":
            x = hog(h.get_hand_img())
        else:
            raise Exception("Invalid Feature Type")

        if training:
            if h.label() != None and h.label() in s.ANNOTATIONS:  # remove hands that are not in the annotations list
                y.append(h.label())
            else:
                raise Exception("No label assigned for training data")

        X.append(x)             # in case of training, don't append to X until we know there is a valid label


    return np.array(X), np.array(y)


def extract_features(img, xtype, n_slices=20, visualise=False, hand="lh"):


    if xtype == "shog":
        return sliced_hog(img, n_slices=n_slices, visualise=visualise)
    elif xtype == "shog2":
        return sliced_hog(img, n_slices=n_slices, visualise=visualise, cells_per_block=(1, 2), block_norm="L1-sqrt")
    elif xtype == "shog3":
        return sliced_hog(img, n_slices=n_slices, visualise=visualise, cells_per_block=(1, 4), block_norm="L1-sqrt")
    elif xtype == "hog":
        return hog(img, visualise=visualise, pixels_per_cell=(16,16), cells_per_block=(1,1))
    elif xtype == "hog2":
        return hog(img, visualise=visualise, pixels_per_cell=(16,16), cells_per_block=(2, 2), block_norm="L1-sqrt")
    elif xtype == "hog3":
        return hog(img, visualise=visualise, pixels_per_cell=(16,16), img_size=(180, 120))
    elif xtype == "hog4":
        return hog(img, visualise=visualise, pixels_per_cell=(16,16), img_size=(180, 120), cells_per_block=(2,2), block_norm="L1-sqrt")
    elif xtype == "hog-new":
        return hog(img, visualise=visualise)
    elif xtype == "hog-new2":
        return hog(img, visualise=visualise, pixels_per_cell=(19,13), cells_per_block=(2,2), block_norm="L1-sqrt", img_size=(190, 130), pad_img=True)
    elif xtype == "hog-new3":
        return hog(img, visualise=visualise, pixels_per_cell=(12,12), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(190, 130), pad_img=False)
    elif xtype == 'honv':
        return honv(img)
    elif xtype == 'honv2':
        return honv(img, pixels_per_cell=(19, 13), num_bins=9, img_size=(190, 130), pad_img=True)
    elif xtype == 'exp1-hog':
        return hog(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(128, 128), pad_img=False)
    elif xtype == 'exp2-hog':
        return hog(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(160, 90), pad_img=False)
    elif xtype == 'exp3-hog':
        return hog(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(189, 130), pad_img=True)
    elif xtype == 'exp4-hog':
        return hog(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(96, 96), pad_img=False)
    elif xtype == 'exp1-honv':
        return honv(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(128, 128), pad_img=False)
    elif xtype == 'exp2-honv':
        return honv(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(160, 90), pad_img=False)
    elif xtype == 'exp3-honv':
        return honv(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(189, 130), pad_img=True)
    elif xtype == 'exp4-honv':
        return honv(img, pixels_per_cell=(8,8), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(96, 96), pad_img=False)
    elif xtype == 'test':
        return hog(img, pixels_per_cell=(4,4), cells_per_block=(3,3), block_norm="L1-sqrt", img_size=(80, 45), pad_img=False)
    else:
        raise Exception("Invalid Feature Type:", xtype)
