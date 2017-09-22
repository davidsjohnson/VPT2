"""
Functions for submodular function maximization
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances as pd

from vpt.common import *

def pairwise_distance(X, Y=None):
    return pd(X,Y, metric='cosine')

class SFO:

    def __init__(self):

        self.dists = np.array([])


    def run(self, data, d=.125, n_max=500):

        if (self.dists.size == 0):
            self.dists = pairwise_distance(data)
        # plt.imshow(self.dists)
        # plt.colorbar()
        # plt.show()
        return self.sfo_greedy_lazy(self.dists, d, n_max = n_max)


    def F(self, W, sset, new_frame, dist_max):
        '''
        Submodular Function - F is the number of frames within the chosen distance, dist_max, to at least one of the frames in ref_set.

        Args:
            W (float[][]):      Similarity matrix for all frames
            sset (int[]):       indexes of frames in the current reference set
            new_frame (int):    index of frame being added to reference set
            dist_max (float):   max distance threshold

        Returns:
            Number of frames covered by reference set
        '''
        test_ref = np.concatenate((sset, [new_frame])).astype(int)      # reference set union with new frame
        return np.sum(np.sum(W[test_ref , :] < dist_max, axis=0 ) > 0)  # count number of frames within thresh per row (axis 0); then count num columns with 1 or more frames within thresh (so we don't count frames twice)


    def sfo_greedy_lazy(self, W, dist_max, n_max=np.inf, tol=1e-6, budget=np.inf, useCB=False, sset=None):
        '''
        Submodular Function Maximization using the greedy lazy algorithm of Krause and Golovin: Submodular Function Maximization

        Args:
            W (float[][]):              pairwise similarity matrix
            dist_max (float):           maximum distance from reference
            n_max (int, optional):      maxumum number of references
            tol(float, optional):       tolerance for convergence
            budget (float, optional):   maximum budget C(sset) < budget (NOT CURRENTLY IMPLEMENTED)
            useCB (bool, optional):     use Cost Benefit approach       (NOT CURRENTLY IMPLEMENTED)
            sset(int[], optional):      initial starting reference set

        Returns:
            A list of reference frame indexes
        '''

        n = W.shape[0]                  # number of frames
        C = np.ones((n,))               #
        V = np.arange(n)                #
        deltas = np.ones((n,))*np.inf   # initialize optimistically

        # start with empty set of specified start set
        if not sset:
            sset = []

        #initialize
        curr_val = 0
        curr_cost = 0

        #keep track of statistics
        eval_num = []
        scores = []

        i = 0
        while len(sset) < n_max:  # build reference set until we reach max size

            best_improve = 0    # Track frame with best improvement to F(sset)
            eval_num.append(0)

            deltas[curr_cost+C > budget] = -np.inf  # change per frame
            order = np.argsort(deltas)[::-1]        # order the frames by delta

            # lazily update improvements
            for test in order:
                if deltas[test] >= best_improve:  # test could be potential best choice

                    eval_num[i] += 1

                    improve = self.F(W, sset, test, dist_max) - curr_val
                    deltas[test] = improve
                    best_improve = max(best_improve, improve)

                elif deltas[test] > -np.inf:
                    break

            argmax = np.argmax(deltas)  # find the best delta

            if deltas[argmax] > tol:
                sset.append(V[argmax])
                curr_val += deltas[argmax]
                curr_cost += C[argmax]
                scores.append(curr_val)

            else:
                break

            i += 1

        return np.array(sset), np.array(scores), eval_num, self.dists



###### Generate Reference Set using Submodular Function Optimization ######

def load_data(folder, shape):

    from vpt.streams.file_stream import FileStream
    from skimage.transform import resize, rescale

    annotations = load_annotations()

    fs = FileStream(folder)
    img_gen = fs.img_generator()

    X = []
    files = []

    for img, fpath in img_gen:
        # img = resize(img, shape)
        img = rescale(img, .25)
        X.append(img.ravel())
        files.append(fpath)

    return np.array(X), np.array(files)

def run():

    folder = "data/posture/p3/"
    annotations = os.path.join(folder, "annotations.txt")
    p = "p3"
    shape = (100, 100)

    X, files = load_data(folder, shape) # What to use for X - scaled down version of image??

    # TODO: Probably need to change these but good for now...
    # dist_thresholds = np.linspace(.025, .03, 4)    # Values from analysis of exercise c
    dist_thresholds = [.0267]    # p4 threshold
    n_maxes = [400]

    sfo = SFO()

    for d in dist_thresholds:
        for n in n_maxes:

            ref_set, scores, eval_num, dists = sfo.run(X, d=d, n_max=n)

            print "\n\nReference Set complete with N =", n
            print "\tDistance Threshold:", str(d)
            print "\tRef Set Length =", str(len(ref_set))
            print "\tRef Set:"
            print "\t\t", ref_set
            # print "\t\t", files[ref_set]

            np.save(p +"_test_reference_set.npy", files[ref_set])

            if len(ref_set) < n:
                break

if __name__ == "__main__":

    run()