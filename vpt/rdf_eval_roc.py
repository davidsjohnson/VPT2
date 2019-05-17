import time
import sys
sys.path.append("./")

from vpt.common import *
import vpt.settings as s
from vpt.streams.compressed_stream import CompressedStream

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve

import vpt.hand_detection.depth_image_features as dif
import vpt.hand_detection.depth_context_features as dcf

# Training Parameters
# Ms = np.array([4,5,6,7])
# radii = np.linspace(.1, .3, 9)
Ms = np.array([5])
radii = np.array([.15])

combined = False
offset_gen = dcf.generate_feature_offsets
feature_gen = dcf.calc_features
refresh = False
n_samples = 750
n_jobs = 1

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

        # for testing_p in training_participants:
        for testing_p in ["p4"]:

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

            X_test_all, y_test_all = np.array([]).reshape(0, rdf_hs.num_features()), np.array([])

            for i, (mask, dmap, fpath) in enumerate(i_gen):

                X_test, y_test = rdf_hs.generate_test_data(dmap, mask)
                X_test_all = np.vstack((X_test_all, X_test))
                y_test_all = np.hstack((y_test_all, y_test))

                if i > 10:
                    break

            print("X Shape", X_test_all.shape)
            print("y Shape", y_test_all.shape)

            y_score = rdf_hs._clf.predict_proba(X_test_all)

            fpr, tpr, _ = roc_curve(y_test_all, y_score[:,2], pos_label=2)

            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

            print("\t\tDone with Participant", testing_p)