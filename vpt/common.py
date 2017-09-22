import sys
import pickle
import os
import re

import cv2
import numpy as np

from vpt.hand_detection.rdf_segmentation_model import RDFSegmentationModel

import vpt.settings as s

def load_hs_model(participant, M, radius, n_samples , refresh, segmentation_model_path):

    if os.path.exists(segmentation_model_path) and refresh != True:
        print "Loading existing hand segmentation model..."
        with open(segmentation_model_path, "r") as f:
            rdf_hs = pickle.load(f)
    else:
        print "Hand segmentation model doesn't exist: %s.  Loading data and training new model..." % (segmentation_model_path)
        rdf_hs = RDFSegmentationModel(M, radius, n_samples)
        rdf_hs.train([os.path.join("data/rdf", participant)])
        with open(segmentation_model_path, "w+") as f:
            pickle.dump(rdf_hs, f)
    return rdf_hs


def load_depthmap(fpath, ftype="bin"):
    ''' Loads and preforms preprocessing steps to a captured image '''

    if fpath[len(fpath)-4:len(fpath)] == ".bin":
        data = np.fromfile(fpath, 'uint16')

    elif fpath[len(fpath)-4:len(fpath)] == ".npy":
        data = np.load(fpath)

    elif fpath[len(fpath)-4:len(fpath)] == ".bmp" or fpath[len(fpath)-4:len(fpath)] == ".jpg":
        data = cv2.imread(fpath)

    else:
        raise Exception('File type not supported:', fpath[len(fpath)-4:len(fpath)])

    if data.shape[0] == 480 * 640:
        data = data.reshape((480, 640))
        # data = cv2.flip(data, 1)
        data = data[80:400, 160:480]
    elif data.shape == (480, 640, 3):
        pass
    else:
        raise Exception("Invalid Data", fpath, data.shape)

    return data


def getFileKey(fpath):

    # add because the file structure of p0 is different than rest which affects loading the
    # correct values from the annotations list
    if s.participant == "p0" or s.participant == "p9":
        reg = '.*(error_[\\d]).*(p[\\d][a-z]).*([\\d]{6})'
    else:
        reg = '.*(p[\\d][a-z]).*([\\d]{6})'

    annot_key = ""
    match = re.match(reg, fpath)
    for m in match.groups():
        annot_key = annot_key + "/" + m

    return annot_key


def load_annotations(annotation_file, debug=False, error=False):

    annotations = {}

    def split_line(line):
        data = line.strip("\n").split("\t")
        try:
            annotations[getFileKey(data[0])] = [data[1], data[2] ]
        except ValueError as e:
            if debug:
                print "Error Loading Annotation:", e

    with open(annotation_file, "rb") as af:
        [split_line(line) for line in af.readlines()]

    return annotations