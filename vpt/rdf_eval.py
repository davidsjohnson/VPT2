import time
import sys
sys.path.append("./")

from vpt.common import *
import vpt.settings as s
from vpt.streams.compressed_stream import CompressedStream

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve

import vpt.hand_detection.depth_image_features as dif
import vpt.hand_detection.depth_context_features as dcf

# # DCF Training Parameters
# Ms = np.array([4,5,6,7])
# radii = np.linspace(.1, .3, 9)

# RDF Training Parameters
Ms = np.array([80, 120, 168, 224])
radii = np.linspace(25000, 150000, 9)

combined = False
offset_gen = dif.generate_feature_offsets
feature_gen = dif.calc_features
refresh = False
n_samples = 750
n_jobs = 5

# Setup some general settings
s.participant = "mix"
s.sensor = "realsense"
training_participants = ["p1", "p2", "p3", "p4", "p6"]
data_folders = {p: "data/rdf/testing/{}".format(p) for p in training_participants}
test_folders = {p: "data/rdf/testing/{}".format(p) for p in training_participants}

base_model_folder = "data/rdf/trainedmodels/"

avg_accs = np.zeros((len(radii), len(Ms)))
avg_Fs = np.zeros((len(radii), len(Ms)))

# Perform leave one out training and testing for each available participant
for idx_rad, radius in enumerate(radii):
    for idx_M, M in enumerate(Ms):

        print("#### Evaluating ####")
        print("M:", M)
        print("Rad:", radius)
        print("Comb:", combined)
        print()

        avg_acc = 0
        avg_F = 0
        total = 0

        for testing_p in training_participants:
        # for testing_p in ["p1"]:

            if not combined:
                seg_model_path = os.path.join(base_model_folder, "{:s}_M{:d}_rad{:0.2f}".format("mixed_no_{}".format(testing_p), M, radius))
            else:
                seg_model_path = os.path.join(base_model_folder, "{:s}_M{:d}_rad{:0.2f}_comb".format("mixed_no_{}".format(testing_p), M, radius))

            print("\t#### Testing Participant {} ####".format(testing_p))
            print("\t\t", end="")

            training_folders = [folder for p, folder in data_folders.items() if p != testing_p]
            test_folder = [test_folders[testing_p]]

            cs = CompressedStream(training_folders)

            print("\t\t", training_folders)
            print("\t\t", test_folders)
            print("\t\tModel Path:", seg_model_path)
            print("\t\tLoading Model...", flush=True)
            print("\t\t", end="")

            model_p = "mixed_no_{}".format(testing_p)
            rdf_hs = load_hs_model(model_p, offset_gen, feature_gen, M, radius, n_samples, n_jobs=n_jobs, refresh=refresh, segmentation_model_path=seg_model_path, ms=cs, combined=combined)

            print("\n\t\t## Testing Model...", flush=True)
            print("\t\t", end="")
            cs_test = CompressedStream(test_folder)
            i_gen = cs_test.img_generator()

            for i, (mask, dmap, fpath) in enumerate(i_gen):

                p_mask = rdf_hs.generate_mask(dmap)

                y_true = np.zeros_like(dmap, dtype="uint8")
                y_true[mask[:, :, 0] > 0] = 1
                y_true[mask[:, :, 1] > 0] = 2
                y_pred = np.zeros_like(dmap, dtype="uint8")
                y_pred[p_mask[:, :, 0] > 0] = 1
                y_pred[p_mask[:, :, 1] > 0] = 2

                accuracy = accuracy_score(y_true=y_true.ravel(), y_pred=y_pred.ravel())
                precision, recall, f, support = precision_recall_fscore_support(y_true=y_true.ravel(), y_pred=y_pred.ravel())

                avg_acc += accuracy
                avg_F += np.average(f)
                total += 1

            print("\t\tDone with Participant", testing_p)



        avg_acc /= total
        avg_F /= total

        print("\tTotal Files = ", total)
        print("\tAvg Acc = ", avg_acc)
        print("\tAvg F = ", avg_F)
        print()
        print()

        avg_accs[idx_rad, idx_M] = avg_acc
        avg_Fs[idx_rad, idx_M] = avg_F

np.savez("rdf_dif_results.npz", accuracy=avg_accs, f_score=avg_Fs)