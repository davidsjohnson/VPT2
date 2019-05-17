from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from vpt.features.features import *
import vpt.utils.image_processing as ip
import vpt.settings as s

import cv2

# IN PROGRESS
#   TODO::::Annotate more data for testing..
#   TODO:::Start processing new data sets - p5 and p6

# START SOON
#   TODO::::Find classification statistics for RDF
#   TODO::::Classification Accuracy based on practice lesson (c, d, e, etc...)

# CLEAN UP / REFACTOR
#   TODO::Fix common and settings files...
#   TODO::Fix imports....

# MAYBE
#   TODO:See if color based background segmentation works better than depth...for training RDF.
#   TODO:Implement Reference Set Selection


def run(participant, hg, feature_type, ds_path=None, fl_path=None, refreshCLF=False, debug=False, n_slices=20):

    print ("Loading Data:", ds_path)

    if not refreshCLF and os.path.exists(ds_path+".npy"):

        dataset = np.load(ds_path+".npy")

        ds_lh = dataset[s.LH]
        split_idx = ds_lh.shape[1]
        split = np.hsplit(ds_lh, [split_idx-1, split_idx])
        X_lh = split[0]
        y_lh = np.squeeze(split[1])

        ds_rh = dataset[s.RH]
        split_idx = ds_rh.shape[1]
        split = np.hsplit(ds_rh, [split_idx-1, split_idx])
        X_rh = split[0]
        y_rh = np.squeeze(split[1])

        training_mask_lh = np.load(ds_path + ".training_lh.npy")
        training_mask_rh = np.load(ds_path + ".training_rh.npy")
        filenames = np.load(ds_path + ".files.npy")

    else:

        print ("Status:",)
        X_lh = []
        y_lh = []
        training_idxs_lh = []

        X_rh = []
        y_rh = []
        training_idxs_rh = []

        filenames = []

        show = False
        for i, (lh, rh) in enumerate(hg.hand_generator(debug)):

            if lh.label() != None and rh.label() != None:

                filenames.append(lh.get_fpath())

                print ("{ LH Label:", lh.label(), end=" ")
                y_lh.append(lh.label())
                X_lh.append(extract_features(lh.get_hand_img(), feature_type, n_slices=n_slices))

                if (participant != "p0" and participant != "p9") and "%ss" % participant in lh.get_fpath():
                    print("HERE: ", lh.get_fpath())
                    training_idxs_lh.append(i)
                elif (participant == "p0" and "p0a" in lh.get_fpath() or "p0b" in lh.get_fpath()) or (participant == "p9" and "p9a" in lh.get_fpath() or "p9b" in lh.get_fpath()):
                    training_idxs_lh.append(i)

                print ("; RH Label:", rh.label(), "}", end=" ")
                y_rh.append(rh.label())
                X_rh.append(extract_features(rh.get_hand_img(), feature_type, n_slices=n_slices))

                if (participant != "p0" and participant != "p9") and "%ss" % participant in rh.get_fpath():
                    training_idxs_rh.append(i)
                elif (participant == "p0" and "p0a" in rh.get_fpath() or "p0b" in rh.get_fpath()) or (participant == "p9" and "p9a" in rh.get_fpath() or "p9b" in rh.get_fpath()):
                    training_idxs_rh.append(i)


                if show:
                    img_norm_lh = ip.normalize(lh.get_hand_img()) * 255
                    img_norm_rh = ip.normalize(rh.get_hand_img()) * 255
                    #
                    cv2.imshow("Left",  img_norm_lh.astype("uint8"))
                    cv2.imshow("Right", img_norm_rh.astype("uint8"))
                    cv2.imshow("Mask LH", lh._mask)
                    cv2.imshow("Mask RH", rh._mask)
                    cv2.moveWindow("Right", 900, 50)
                    cv2.moveWindow("Mask LH", 200, 50)
                    cv2.moveWindow("Mask RH", 200, 500)

                    wait_char = cv2.waitKey(1)
                    if wait_char == ord('q'):
                        cv2.destroyAllWindows()
                        exit()
                    elif wait_char == 27:
                        show=False
                        cv2.destroyAllWindows()

            print (i,end=" ")

        cv2.destroyAllWindows()

        print ()
        print ("Retrieved hands")
        print ("Generating and saving dataset...")

        X_lh = np.array(X_lh)
        y_lh = np.array(y_lh)
        X_rh = np.array(X_rh)
        y_rh = np.array(y_rh)

        # plt.plot(y_lh)
        # plt.show()

        training_mask_lh = np.zeros((len(X_lh),), dtype=bool)
        training_mask_lh[training_idxs_lh] = True

        training_mask_rh = np.zeros((len(X_rh),), dtype=bool)
        training_mask_rh[training_idxs_rh] = True

        # save datasets
        print ("LH:", np.array(X_lh).shape, np.array(y_lh).shape, training_mask_lh.shape, training_mask_lh[training_mask_lh==True].shape)
        print ("RH:", np.array(X_rh).shape, np.array(y_rh).shape, training_mask_rh.shape, training_mask_rh[training_mask_lh==True].shape)

        print()

        filenames = np.array(filenames)

        np.save(ds_path, [np.hstack((X_lh, np.expand_dims(y_lh, 1))), np.hstack((X_rh, np.expand_dims(y_rh, 1)))])
        np.save(ds_path + ".training_lh.npy", training_mask_lh)
        np.save(ds_path + ".training_rh.npy", training_mask_rh)
        np.save(ds_path + ".files.npy", filenames)

    # plt.plot(y_lh[training_mask_lh])
    # plt.show()

    print ("Data Set Generated")
    print ()
    print ("Starting Classification...")

    clf_lh = LinearSVC(class_weight="balanced", C=1, dual=False)
    clf_rh = LinearSVC(class_weight="balanced", C=1, dual=False)

    X_train_lh, y_train_lh = SMOTE(kind='svm').fit_sample(X_lh[training_mask_lh], y_lh[training_mask_lh])
    X_test_lh, y_test_lh = X_lh[~training_mask_lh], y_lh[~training_mask_lh]

    X_train_rh, y_train_rh = SMOTE(kind='svm').fit_sample(X_rh[training_mask_rh], y_lh[training_mask_rh])
    X_test_rh, y_test_rh = X_rh[~training_mask_rh], y_rh[~training_mask_rh]


    X_lh, y_lh = SMOTE(kind='svm').fit_sample(X_lh, y_lh)
    X_rh, y_rh = SMOTE(kind='svm').fit_sample(X_rh, y_rh)

    # #######  Test Left Hand
    ########################
    clf_lh.fit(X_lh, y_lh)
    print ("\tLeft Hand Score:", clf_lh.score(X_lh, y_lh))
    print ("\tLeft Hand CV Score:", cross_val_score(clf_lh, X_lh, y_lh, cv=5))

    if training_mask_lh.max() > 0:

        clf_lh.fit(X_train_lh, y_train_lh)
        print ("\tLeft Hand Score (Static):", clf_lh.score(X_test_lh, y_test_lh))

        lh_preds = clf_lh.predict(X_test_lh)
        lh_truth = y_test_lh
        print (confusion_matrix(lh_truth, lh_preds))

        # with open(ds_path+".clf_lh.npy", "w+") as f:
        #     pickle.dump(clf_lh, f)

        # with open(ds_path + ".p_results_lh.txt", "w+") as f:
        #     f.write("Filename\t\tTruth\tPred\n")
        #     for i, filename in enumerate(filenames[~training_mask_lh]):
        #         f.write("%s\t%i\t%i\n" % (filename, y_lh[~training_mask_lh][i], lh_preds[i]))
        #
        #
        # X_lh_test_0 = X_lh[~training_mask_lh][np.where(y_lh[~training_mask_lh] == 0)]
        # X_lh_test_1 = X_lh[~training_mask_lh][np.where(y_lh[~training_mask_lh] == 1)]
        # X_lh_test_2 = X_lh[~training_mask_lh][np.where(y_lh[~training_mask_lh] == 2)]
        #
        # X_lh_0 = X_lh[training_mask_lh][np.where(y_lh[training_mask_lh] == 0)]
        # X_lh_1 = X_lh[training_mask_lh][np.where(y_lh[training_mask_lh] == 1)]
        # X_lh_2 = X_lh[training_mask_lh][np.where(y_lh[training_mask_lh] == 2)]

        # np.save("data/analysis/" + ds_path + "_test_Xlh_0", X_lh_test_0)
        # np.save("data/analysis/" + ds_path + "_test_Xlh_1", X_lh_test_1)
        # np.save("data/analysis/" + ds_path + "_test_Xlh_2", X_lh_test_2)
        #
        # np.save("data/analysis/" + ds_path + "_Xlh_0", X_lh_0)
        # np.save("data/analysis/" + ds_path + "_Xlh_1", X_lh_1)
        # np.save("data/analysis/" + ds_path + "_Xlh_2", X_lh_2)

    ########  Test Right Hand
    ########################
    clf_rh.fit(X_rh, y_rh)
    print ("\tRight Hand Score:", clf_rh.score(X_rh, y_rh))
    print ("\tRight Hand CV Score:", cross_val_score(clf_rh, X_rh, y_rh, cv=5))
    if training_mask_rh.max() > 0:

        clf_rh.fit(X_train_rh, y_train_rh)
        print ("\tRight Hand Score (Static):", clf_rh.score(X_test_rh, y_test_rh))

        rh_preds = clf_rh.predict(X_test_rh)
        rh_truth = y_test_rh
        print (confusion_matrix(rh_truth, rh_preds))

        # with open(ds_path+".clf_rh.npy", "w+") as f:
        #     pickle.dump(clf_rh, f)

        # with open(ds_path + ".p_results_rh.txt", "w+") as f:
        #     f.write("Filename\t\tTruth\tPred\n")
        #     for i, filename in enumerate(filenames[~training_mask_rh]):
        #         f.write("%s\t%i\t%i\n" % (filename, y_rh[~training_mask_rh][i], rh_preds[i]))
        #
        # X_rh_test_0 = X_rh[~training_mask_rh][np.where(y_rh[~training_mask_rh] == 0)]
        # X_rh_test_1 = X_rh[~training_mask_rh][np.where(y_rh[~training_mask_rh] == 1)]
        # X_rh_test_2 = X_rh[~training_mask_rh][np.where(y_rh[~training_mask_rh] == 2)]
        #
        # X_rh_0 = X_rh[training_mask_rh][np.where(y_rh[training_mask_rh] == 0)]
        # X_rh_1 = X_rh[training_mask_rh][np.where(y_rh[training_mask_rh] == 1)]
        # X_rh_2 = X_rh[training_mask_rh][np.where(y_rh[training_mask_rh] == 2)]
        #
        # np.save("data/analysis/" + ds_path + "_test_Xrh_0", X_rh_test_0)
        # np.save("data/analysis/" + ds_path + "_test_Xrh_1", X_rh_test_1)
        # np.save("data/analysis/" + ds_path + "_test_Xrh_2", X_rh_test_2)
        #
        # np.save("data/analysis/" + ds_path + "_Xrh_0", X_rh_0)
        # np.save("data/analysis/" + ds_path + "_Xrh_1", X_rh_1)
        # np.save("data/analysis/" + ds_path + "_Xrh_2", X_rh_2)

    print ()





def run_with_rdf(participant, ftype="bin", refreshHD=False, refreshCLF=False, Ms = (3,), radii = (.3,),  n_samples = 500, feature_types = ("shog",)):


    # dataset parameters
    folder = os.path.join("data/posture", participant)

    for M in Ms:
        for radius in radii:
            for feature_type in feature_types:

                ds_path = "%s_M%i_rad%0.1f_%s%s" % (participant, M, radius, feature_type, s.note)
                print (ds_path)
                ds_path = os.path.join(folder, ds_path)

                # segmentation_model_path = "vpt/hand_detection/model/%s_M%i_rad%0.1f" % (participant if participant != "p9" else "p1", M, radius)  # added p9 check for testing with smaller dataset
                segmentation_model_path = "data/rdf/trainedmodels/%s_M%i_rad%0.1f" % (participant, M, radius)

                annotation_file = os.path.join(folder, "annotations.txt")

                print ("\nRunning Classification with following parameters:")
                print()
                print ("\tData set Path:", ds_path)
                print ("\tAnnotation File:", annotation_file)
                print()
                print ("\tDCF M:", M)
                print ("\tDCF Radius:", radius)
                print ("\tDCF Seg Model:", segmentation_model_path)
                print()
                print ("\tFeatures:", feature_type)
                print()
                print()

                print ("Initializing Hand Pipeline")

                annotations = load_annotations(annotation_file)
                print ("\tAnnotations Loaded")

                # generate or load model
                rdf_hs = load_hs_model(s.participant, M, radius, n_samples , refreshHD, segmentation_model_path)

                fs = FileStream(folder, ftype, annotations=annotations, ignore=True)
                hd = HandDetector(rdf_hs)
                hg = HandGenerator(fs, hd, annotations)
                print ("\tHand Generator Loaded")
                run(participant, hg, feature_type, ds_path, None, refreshCLF, debug=True)



def run_with_bsub(participant, ftype="bin", refreshHD=False, refreshCLF=True):

    # Classification/feature parameters
    feature_types = ["shog"]

    # dataset parameters
    folder = os.path.join("data", participant)

    background_folder = "data/%s/background_model" % (participant)

    for feature_type in feature_types:
        segmentation_model = "%sTestModel" % (participant)

        ds_path = "%s_%s_bsub" % (participant, feature_type)
        ds_path = os.path.join(folder, ds_path)

        annotation_file = os.path.join(folder, "annotations.txt")

        print ("\nRunning Classification with following parameters:")
        print ("\tHand Detection via Background Subtraction")
        print()
        print ("\tData set Path:", ds_path)
        print ("\tAnnotation File:", annotation_file)
        #                                 TODO: Start Re-viewing papers
        #                                 TODO: Start considering speed vs. accuracy...! (and can I get RDF implmeneted in at least near realtime
        print()
        print ("\tFeatures:", feature_type)
        print()
        print()

        print ("Initializing Hand Pipeline")

        annotations = load_annotations(annotation_file)
        print ("\tAnnotations Loaded")

        fs = FileStream(folder, ftype, annotations, ignore=True)
        hs = SubSegmentationModel(segmentation_model)
        hs.initialize(background_folder, varThreshold=.3)
        hd = HandDetector(hs)
        hg = HandGenerator(fs, hd, annotations)
        print ("\tHand Generator Loaded")
        run(participant, hg, feature_type, ds_path, None, refreshCLF, debug=True)


if __name__ == "__main__":

    # Participants
    #   p0 - dav1
    #   p9 - isa2
    #   p1-p6 - students (realsense)

    s.participant = "p1"
    s.sensor = "realsense"  #TODO::::Implement RS sensor distortion
    s.note = ""

    refreshHD = False
    refreshCLF = True

    run_with_rdf(s.participant, ftype="bin", refreshHD=refreshHD, refreshCLF=refreshCLF, Ms=(3,), radii=(.3,), feature_types=("hog",))           # ready to run with new params....
    # run_with_rdf("p9", ftype="bin", refresh=True)  # ready to run with new params....
    # run_with_bsub(s.participant, ftype="bin", refreshHD=refreshHD, refreshCLF=refreshCLF)
    os.system('say "VPT2 Classification Completed"')