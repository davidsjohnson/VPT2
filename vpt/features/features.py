from vpt.common import *

from skimage.transform import resize
from skimage.feature import hog as skhog

from vpt.hand_detection.hand_generator import *
import vpt.settings as s

def hog(img, visualise=False, cells_per_block=(1,1), block_norm="L1-sqrt", img_size=(180,180)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = resize(img, img_size)
        return skhog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=cells_per_block, block_norm=block_norm, visualise=visualise)


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
        return hog(img, visualise=visualise)
    elif xtype == "hog2":
        return hog(img, visualise=visualise, cells_per_block=(2, 2), block_norm="L1-sqrt")
    elif xtype == "hog3":
        return hog(img, visualise=visualise, img_size=(180, 120))
    elif xtype == 'shonv':
        return shonv(img)
    else:
        raise Exception("Invalid Feature Type:", xtype)
