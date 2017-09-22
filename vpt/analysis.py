import matplotlib.pyplot as plt
import cv2

from vpt.common import *
from vpt.hand_detection.rdf_segmentation_model import *
from vpt.hand_detection.hand_detector import *
from vpt.features.features import *
import vpt.settings as s
import vpt.utils.image_processing as ip

def load_results(results_file):

    files = []
    y = []
    y_ = []

    with open(results_file, 'r') as f:
        for i, line in enumerate(f):
            if i > 0:
                line = line.split()
                if len(line) == 3:
                    files.append(line[0])
                    y.append(line[1])
                    y_.append(line[2])
                else:
                    raise AttributeError("Invalid line in results files")


    return files, y, y_

def analyze(participant, M, radius, f_type, refresh=False):

    if refresh:

        annotations = load_annotations("data/p1/annotations.txt")

        segmentation_model_path = "%s_M%i_rad%0.1f" % (participant, M, radius)
        rdf_hs = load_hs_model(participant, M, radius, 500, False, segmentation_model_path)
        hd = HandDetector(rdf_hs)

        fs = FileStream("data/p1", 'bin')
        hg = HandGenerator(fs, hd, annotations)

        X_lh_0 = []
        X_lh_1 = []
        X_lh_2 = []

        X_rh_0 = []
        X_rh_1 = []
        X_rh_2 = []

        vis_lh_0 = np.zeros((180, 240))
        vis_lh_1 = np.zeros((180, 240))
        vis_lh_2 = np.zeros((180, 240))

        vis_rh_0 = np.zeros((180, 240))
        vis_rh_1 = np.zeros((180, 240))
        vis_rh_2 = np.zeros((180, 240))

        for i, (lh, rh) in enumerate(hg.hand_generator(debug=True)):

            print i, ": LH y -", lh.label(), " |  RH y - ", rh.label(), " ||||| ",

            img_norm_lh = (ip.normalize(lh.get_hand_img()) * 255).astype("uint8")
            features_lh, vis_lh = extract_features(img_norm_lh, f_type, n_slices=20, visualise=True)
            if lh.label() == 0 or lh.label() == 3:
                X_lh_0.append(features_lh)
                vis_lh_0 += vis_lh

            elif lh.label() == 1 or lh.label() == 4:
                X_lh_1.append(features_lh)
                vis_lh_1 += vis_lh

            elif lh.label() == 2 or lh.label() == 5:
                X_lh_2.append(features_lh)
                vis_lh_2 += vis_lh

            img_norm_rh = (ip.normalize(rh.get_hand_img()) * 255).astype("uint8")
            features_rh, vis_rh = extract_features(img_norm_rh, f_type, n_slices=20, visualise=True)
            if rh.label() == 0 or rh.label() == 3:
                X_rh_0.append(features_rh)
                vis_rh_0 += vis_rh

            elif rh.label() == 1 or rh.label() == 4:
                X_rh_1.append(features_rh)
                vis_rh_1 += vis_rh

            elif rh.label() == 2 or rh.label() == 5:
                X_rh_2.append(features_rh)
                vis_rh_2 += vis_rh
        print

        X_lh_0 = np.array(X_lh_0)
        X_lh_1 = np.array(X_lh_1)
        X_lh_2 = np.array(X_lh_2)

        X_rh_0 = np.array(X_rh_0)
        X_rh_1 = np.array(X_rh_1)
        X_rh_2 = np.array(X_rh_2)

        avg_vis_lh_0 = vis_lh_0 / (X_lh_0.shape[0])
        avg_vis_lh_1 = vis_lh_1 / (X_lh_1.shape[0])
        avg_vis_lh_2 = vis_lh_2 / X_lh_2.shape[0]

        avg_vis_rh_0 = vis_rh_0 / (X_rh_0.shape[0])
        avg_vis_rh_1 = vis_rh_1 / (X_rh_1.shape[0])
        avg_vis_rh_2 = vis_rh_2 / X_rh_2.shape[0]

        np.save("data/analysis/p1_M3_rad0.3_shog_lh_0", X_lh_0)
        np.save("data/analysis/p1_M3_rad0.3_shog_lh_1", X_lh_1)
        np.save("data/analysis/p1_M3_rad0.3_shog_lh_2", X_lh_2)

        np.save("data/analysis/p1_M3_rad0.3_shog_rh_0", X_rh_0)
        np.save("data/analysis/p1_M3_rad0.3_shog_rh_1", X_rh_1)
        np.save("data/analysis/p1_M3_rad0.3_shog_rh_2", X_rh_2)

        np.save("data/analysis/p1_M3_rad0.3_shog_lh_0_vis", avg_vis_lh_0)
        np.save("data/analysis/p1_M3_rad0.3_shog_lh_1_vis", avg_vis_lh_1)
        np.save("data/analysis/p1_M3_rad0.3_shog_lh_2_vis", avg_vis_lh_2)

        np.save("data/analysis/p1_M3_rad0.3_shog_rh_0_vis", avg_vis_rh_0)
        np.save("data/analysis/p1_M3_rad0.3_shog_rh_1_vis", avg_vis_rh_1)
        np.save("data/analysis/p1_M3_rad0.3_shog_rh_2_vis", avg_vis_rh_2)

    else:

        X_lh_0 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_Xlh_0.npy")
        X_lh_1 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_Xlh_1.npy")
        X_lh_2 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_Xlh_2.npy")

        X_rh_0 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_Xrh_0.npy")
        X_rh_1 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_Xrh_1.npy")
        X_rh_2 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_Xrh_2.npy")

        X_lh_test_0 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_test_Xlh_0.npy")
        X_lh_test_1 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_test_Xlh_1.npy")
        X_lh_test_2 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_test_Xlh_2.npy")

        X_rh_test_0 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_test_Xrh_0.npy")
        X_rh_test_1 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_test_Xrh_1.npy")
        X_rh_test_2 = np.load("data/analysis/data/p1/p1_M3_rad0.3_shog_test_Xrh_2.npy")

        # avg_vis_lh_0 = np.load("data/analysis/p1_M3_rad0.3_shog_lh_0_vis.npy")
        # avg_vis_lh_1 = np.load("data/analysis/p1_M3_rad0.3_shog_lh_1_vis.npy")
        # avg_vis_lh_2 = np.load("data/analysis/p1_M3_rad0.3_shog_lh_2_vis.npy")
        #
        # avg_vis_rh_0 = np.load("data/analysis/p1_M3_rad0.3_shog_rh_0_vis.npy")
        # avg_vis_rh_1 = np.load("data/analysis/p1_M3_rad0.3_shog_rh_1_vis.npy")
        # avg_vis_rh_2 = np.load("data/analysis/p1_M3_rad0.3_shog_rh_2_vis.npy")


    print "X LH 0:", X_lh_0.shape
    print "X LH 1:", X_lh_1.shape
    print "X LH 2:", X_lh_2.shape
    print
    print "X RH 0:", X_rh_0.shape
    print "X RH 1:", X_rh_1.shape
    print "X RH 2:", X_rh_2.shape

    #
    # # Left Hand
    # plt.figure(1)
    #
    # plt.subplot(321)
    # plt.axis([0,1.0, 0, 100])
    # plt.hist(X_lh_0.mean(axis=0).ravel())
    # plt.subplot(322)
    # plt.imshow(avg_vis_lh_0)
    #
    # plt.subplot(323)
    # plt.axis([0, 1.0, 0, 100])
    # plt.hist(X_lh_1.mean(axis=0).ravel())
    # plt.subplot(324)
    # plt.imshow(avg_vis_lh_1)
    #
    # plt.subplot(325)
    # plt.axis([0, 1.0, 0, 100])
    # plt.hist(X_lh_2.mean(axis=0).ravel())
    # plt.subplot(326)
    # plt.imshow(avg_vis_lh_2)
    #
    # # Right Hand
    # plt.figure(2)
    #
    # plt.subplot(321)
    # plt.axis([0, 1.0, 0, 100])
    # plt.hist(X_rh_0.mean(axis=0).ravel())
    # plt.subplot(322)
    # plt.imshow(avg_vis_rh_0)
    #
    # plt.subplot(323)
    # plt.axis([0, 1.0, 0, 100])
    # plt.hist(X_rh_1.mean(axis=0).ravel())
    # plt.subplot(324)
    # plt.imshow(avg_vis_rh_1)
    #
    # # plt.subplot(325)
    # # plt.axis([0, 1.0, 0, 100])
    # # plt.hist(X_rh_2.mean(axis=0).ravel())
    # # plt.subplot(326)
    # # plt.imshow(avg_vis_rh_2)


    avg_X_lh_0 = X_lh_0.mean(axis=0)
    avg_X_lh_1 = X_lh_1.mean(axis=0)
    avg_X_lh_2 = X_lh_2.mean(axis=0)

    avg_X_rh_0 = X_rh_0.mean(axis=0)
    avg_X_rh_1 = X_rh_1.mean(axis=0)
    avg_X_rh_2 = X_rh_2.mean(axis=0)

    avg_X_lh_test_0 = X_lh_test_0.mean(axis=0)
    avg_X_lh_test_1 = X_lh_test_1.mean(axis=0)
    avg_X_lh_test_2 = X_lh_test_2.mean(axis=0)

    avg_X_rh_test_0 = X_rh_test_0.mean(axis=0)
    avg_X_rh_test_1 = X_rh_test_1.mean(axis=0)
    avg_X_rh_test_2 = X_rh_test_2.mean(axis=0)


    # LH - Train Feature Averages
    plt.figure(1)

    plt.subplot(311)
    x = np.arange(avg_X_lh_0.shape[0])
    plt.bar(x, avg_X_lh_0)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(312)
    x = np.arange(avg_X_lh_1.shape[0])
    plt.bar(x, avg_X_lh_1)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(313)
    x = np.arange(avg_X_lh_2.shape[0])
    plt.bar(x, avg_X_lh_2)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    # RH - Train Feature Averages
    plt.figure(2)

    plt.subplot(311)
    x = np.arange(avg_X_rh_0.shape[0])
    plt.bar(x, avg_X_rh_0)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(312)
    x = np.arange(avg_X_rh_1.shape[0])
    plt.bar(x, avg_X_rh_1)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(313)
    x = np.arange(avg_X_rh_2.shape[0])
    plt.bar(x, avg_X_rh_2)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])


    # LH - Test Data Features
    plt.figure(3)

    plt.subplot(311)
    x = np.arange(avg_X_lh_test_0.shape[0])
    plt.bar(x, avg_X_lh_test_0)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(312)
    x = np.arange(avg_X_lh_test_1.shape[0])
    plt.bar(x, avg_X_lh_test_1)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(313)
    x = np.arange(avg_X_lh_test_2.shape[0])
    plt.bar(x, avg_X_lh_test_2)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    # RH - Test Data Features
    plt.figure(4)

    plt.subplot(311)
    x = np.arange(avg_X_rh_test_0.shape[0])
    plt.bar(x, avg_X_rh_test_0)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    plt.subplot(312)
    x = np.arange(avg_X_rh_test_1.shape[0])
    plt.bar(x, avg_X_rh_test_1)
    plt.yscale('log')
    plt.axis([0, 160, .0001, 10])

    # plt.subplot(313)
    # x = np.arange(avg_X_rh_2.shape[0])
    # plt.bar(x, avg_X_rh_2)
    # # plt.yscale('log')
    # plt.axis([0, 160, .1])

    plt.show()

# def view_error(participant, M, radius, f_type, results_file):
#
#     segmentation_model_path = "%s_M%i_rad%0.1f" % (participant, M, radius)
#     rdf_hs = load_hs_model(participant, M, radius, 500, False, segmentation_model_path)
#     hd = HandDetector(rdf_hs)
#
#     files, y, y_ = load_results(results_file)
#
#     for f, l, p in zip(files, y, y_):
#
#         if l == p:
#             depth_map = load_depthmap(f)
#             img = (ip.normalize(depth_map)*255).astype("uint8")
#
#             lh, rh = hd.detect_hands(depth_map, f, (s.ANNOTATIONS[int(l)], s.ANNOTATIONS[int(l)]))
#
#             img_norm_lh = (ip.normalize(lh.get_hand_img()) * 255).astype("uint8")
#
#             f = extract_features(lh.get_hand_img(), f_type, n_slices=20)



if __name__ == "__main__":
    s.participant = "p1"
    s.sensor = "realsense"
    s.note = ""
    M = 3
    radius = .3
    f_type = "shog"

    refresh = False

    analyze(s.participant, M, radius, f_type, refresh=refresh)
