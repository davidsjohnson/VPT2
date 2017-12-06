import sys
sys.path.append("./")

import argparse
import datetime

from sklearn.model_selection import train_test_split

from vpt.features.features import *
from vpt.features.autoencoder import CAE
import vpt.utils.image_processing as ip
import vpt.settings as s


def generate_dataset(hg, size=(120, 96)):

    h_gen = hg.hand_generator(debug=True)
    X_lh, X_rh = [], []

    for lh, rh in h_gen:
        img_lh = ip.normalize(resize(lh.get_hand_img(), size))
        img_rh = ip.normalize(resize(rh.get_hand_img(), size))
        X_lh.append(np.expand_dims(img_lh, axis=2))
        X_rh.append(np.expand_dims(img_rh, axis=2))

    return np.array(X_lh), np.array(X_rh)



def main(participant, folder, annotation_file, n_epochs, batch_size,
         sensor="realsense", ftype="bin", M=3, radius=.3, n_samples=500, refreshHD=False, refreshData=False):

    s.participant = participant
    s.sensor = sensor

    annotations = load_annotations(annotation_file)
    segmentation_model_path = "data/rdf/trainedmodels/%s_M%i_rad%0.1f" % (s.participant, M, radius)

    rdf_hs = load_hs_model(s.participant, M, radius, n_samples, refreshHD, segmentation_model_path)

    fs = FileStream(folder, ftype, annotations=annotations, ignore=True)
    hd = HandDetector(rdf_hs)
    hg = HandGenerator(fs, hd, annotations)


    print("Generating Hand Dataset...")

    if refreshData:
        print("\tGenerating new hand data")
        X_lh, X_rh = generate_dataset(hg)
        np.save("data/posture/all/X_lh.npy", X_lh)
        np.save("data/posture/all/X_rh.npy", X_rh)
    else:
        print("\tLoading existing data")
        X_lh, X_rh = np.load("data/posture/all/X_lh.npy"), np.load("data/posture/all/X_rh.npy")

    print("Done")
    print("\tLH:", X_lh.shape)
    print("\tRH:", X_rh.shape)
    print()

    print("Training CAEs...")
    X_train_lh, X_test_lh, X_train_rh, X_test_rh = train_test_split(X_lh, X_rh, test_size=.1)

    cae_lh, cae_rh = CAE(img_shape=X_lh[0].shape), CAE(img_shape=X_rh[0].shape)
    cae_lh.fit(X_train_lh, X_test_lh, epochs=n_epochs, batch_size=batch_size)
    cae_rh.fit(X_train_rh, X_test_rh, epochs=n_epochs, batch_size=batch_size)
    print("Done")
    print()

    print("Saving CAEs...")
    try:
        cae_lh.save("data/cae", prefix="lh_{}_".format(datetime.date.today().strftime("%Y%m%d")))
        cae_rh.save("data/cae", prefix="rh_{}_".format(datetime.date.today().strftime("%Y%m%d")) )
    except Exception as e:
        print("Issue Saving: ", e)
    print("Done")
    print()

if __name__ == "__main__":

    # Configure Command Line arguments
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder for image feature extraction.")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-p", "--participant", type=str, help="Participant Identifier",
                          metavar="<participant>", required=True)
    required.add_argument("-f", "--folder", type=str, help="Folder containing the participant recording",
                          metavar="<video folder>", required=True)
    required.add_argument("-a", "--annotations", type=str, help="File containing participant annotations",
                          metavar="<annotations file>", required=True)
    required.add_argument("-e", "--epochs", type=int, help="The number of epochs the training should run for",
                          metavar="<num epochs>", required=True)
    required.add_argument("-b", "--batchsize", type=int, help="The number of images per batch", metavar="<batch size>",
                          required=True)

    args = parser.parse_args()

    # currently using default RDF parameters
    main(args.participant, args.folder, args.annotations, n_epochs=args.epochs, batch_size=args.batchsize, refreshData=True)