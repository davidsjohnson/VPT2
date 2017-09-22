from vpt.common import *
import matplotlib.pyplot as plt

import depth_context_features as dcf


class RDFModelError(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)


class RDFParamsError(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)


class RDFSegmentationModelOLD():

    def __init__(self, model_name):

        model_fname = "vpt/hand_detection/model_old/"+model_name+"_clf.pkl"
        params_fname = "vpt/hand_detection/model_old/"+model_name+"_params.pkl"

        try:
            with open(model_fname) as f:
                self.clf = pickle.load(f)
        except IOError as e:
            raise RDFModelError("RDF Model classifier doesn't exist: %s. Train a model_old and try again" % (model_name))

        try:
            with open(params_fname) as f:
                self.params = pickle.load(f)
        except IOError as e:
            raise RDFParamsError("RDF Parameters don't exist. Create parameters file and try again")

        self.offsets = dcf.generate_feature_offsets(self.params.M, self.params.radius)


    def generate_mask(self, depth_map):

        mask_shape = (depth_map.shape[0], depth_map.shape[1], 3)
        mask = np.zeros(mask_shape, dtype="uint8")

        all_features1 = dcf.calc_features(depth_map, self.offsets)

        # plt.hist(all_features1)
        # plt.show()

        p = self.clf.predict(np.array(all_features1).squeeze())
        p = p.reshape(depth_map.shape)

        mask[:, :, LH][p == LH_LBL] = 255
        mask[:, :, RH][p == RH_LBL] = 255

        return mask