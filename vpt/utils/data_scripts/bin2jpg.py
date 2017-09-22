import os
import numpy as np
import cv2
import shutil

from vpt.streams.file_stream import *
from vpt.streams.refset_stream import *
import vpt.utils.image_processing as ip

import matplotlib.pyplot as plt

def bin2jpg(fs):

    fgen = fs.img_generator()

    for img, fname in fgen:
        img = ip.normalize2(img) * 255
        img = img.astype('uint8')

        folder = fname[0:-10]
        folder = os.path.join(folder, "masks")
        file = fname[-10:-4] + ".jpg"
        fullpath = os.path.join(folder, file)

        if not os.path.exists(folder):
            os.mkdir(folder)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(fullpath, img)


def create_masks(fs):

    fgen = fs.img_generator()

    lower_red = np.array([0, 200, 200])
    upper_red = np.array([2, 255, 255])

    lower_blue = np.array([110, 200, 200])
    upper_blue = np.array([120, 255, 255])

    for img, fname in fgen:

        temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        maskLH = cv2.inRange(temp, lower_blue, upper_blue)
        maskRH = cv2.inRange(temp, lower_red, upper_red)

        maskLH[:180, :] = 0
        maskRH[:180, :] = 0

        mask = np.zeros_like(img, dtype="uint8")
        mask[:,:,0] = maskLH
        mask[:,:,1] = maskRH

        folder = fname[0:-10]
        newfolder = os.path.join(folder, "masks")
        mk_file = fname[-10:-4] + "_mask.npy"

        og_file = fname[-10:-4] + ".bin"

        maskpath = os.path.join(newfolder, mk_file)

        # plt.imshow(mask)
        # plt.show()

        np.save(maskpath, mask)


def load_masks(fs):

    base_folder = "data/testdata"

    fgen = fs.img_generator()
    for mask, fpath in fgen:

        temp = fpath.split("/")
        participant = temp[2]
        exercise = temp[3]
        file_num = temp[-1][:6]

        og_path = os.path.join(base_folder, participant, exercise, file_num+".bin")

        dmap = load_depthmap(og_path)
        dmap_rgb = (ip.normalize(dmap) * 255).astype('uint8')
        dmap_rgb = cv2.cvtColor(dmap_rgb, cv2.COLOR_GRAY2RGB)

        dmap_rgb[:, :, 0][np.where(mask[:, :, 0] == 255)] = 255
        dmap_rgb[:, :, 1][np.where(mask[:, :, 0] == 255)] = 0
        dmap_rgb[:, :, 2][np.where(mask[:, :, 0] == 255)] = 0

        dmap_rgb[:, :, 1][np.where(mask[:, :, 1] == 255)] = 255
        dmap_rgb[:, :, 0][np.where(mask[:, :, 1] == 255)] = 0
        dmap_rgb[:, :, 2][np.where(mask[:, :, 1] == 255)] = 0


        # lh_dmap = np.bitwise_and(dmap, mask[:, :, 0])
        # rh_dmap = np.bitwise_and(dmap, mask[:, :, 1])

        plt.figure()
        plt.subplot(131)
        plt.imshow(mask)
        plt.subplot(132)
        plt.imshow(dmap_rgb)
        plt.subplot(133)
        plt.imshow(dmap)
        plt.show()


def retrieve_color(filelist):

    basefolder = "/Volumes/SILVER/vpt_data/"
    newfolder = "data/posture"


    for f in filelist:

        temp = f.split("/")
        participant = temp[2]
        exercise = temp[3]
        file = temp[4].strip("bin")
        file += "bmp"

        fullpath = os.path.join(basefolder, participant, exercise, "color", file)
        newpath = os.path.join(newfolder, participant, exercise, "masks")
        newfile = os.path.join(newpath, file)

        if not os.path.exists(newpath):
            os.mkdir(newpath)

        shutil.copy(fullpath, newfile)

        # print fullpath, newfile



if __name__ == "__main__":

    # folder = "data/p1/p1a"
    # fs = FileStream(folder)
    #
    # bin2bmp(fs)

    folder = "data/testdata/p4/p4a/masks"
    fs = FileStream(folder, ftype="jpg")
    create_masks(fs)

    folder = "data/testdata/p4/p4a/masks/masks"
    fs = FileStream(folder, ftype="npy")
    load_masks(fs)

    # file = "p4_test_reference_set.npy"
    # fpaths = np.load(file)
    #
    # retrieve_color(fpaths)

    #TODO:  FIND A REFERENCE SET TO USE