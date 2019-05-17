import pickle

from vpt.common import *

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline



def load_data(testing_p, M, radius, feature_type, data_type):
    base = "data/posture/extracted/"
    data_path = os.path.join(base, "{}_M{}_rad{:0.2f}_{}_".format(testing_p, M, radius, feature_type))
    data = np.load(data_path + data_type + "_data_combined.npz")
    return data


def train_test_by_participant(X, y, f, participant):
    # split data by participant
    r = re.compile(participant)
    vmatch = np.vectorize(lambda x: bool(r.search(x)))
    val_p = vmatch(f)

    X_train, y_train, f_train = X[~val_p], y[~val_p], f[~val_p]
    X_test, y_test, f_test    = X[val_p],  y[val_p],  f[val_p]

    return X_train, y_train, f_train, X_test, y_test, f_test


def remove_static(X, y, f):
    # remove p#s data
    r = re.compile('p[\d]s')
    vmatch = np.vectorize(lambda x: bool(r.search(x)))
    rem_static = vmatch(f)

    return X[~rem_static], y[~rem_static], f[~rem_static]


def data_report(X, y, f):
    print("\t\t\tX:", X.shape)
    print("\t\t\ty:", y.shape, np.unique(y, return_counts=True))
    print("\t\t\tFiles:", f.shape)


def cross_validation(pipeline, X, y, f, participants, rem_static=0, verbose=0):
    print("\tPerforming Cross Validation for pipeline\n\t", pipeline)

    acc_avg = 0
    fsc_avg = 0

    for i, p in enumerate(participants):
        X_train, y_train, f_train, X_test, y_test, f_test = train_test_by_participant(X, y, f, p)


        if rem_static > 0:
            X_test, y_test, f_test = remove_static(X_test, y_test, f_test)

        if rem_static > 1:
            X_train, y_train, f_train = remove_static(X_train, y_train, f_train)


        print()
        print("\t\t###### CV {} - Testing {} ######".format(i, p))

        if verbose > 1:
            print("\t\tTraining Data:")
            data_report(X_train, y_train, f_train)
            print()
            print("\t\tTesting Data:")
            data_report(X_test, y_test, f_test)
            print()

        print("\t\tTraining Classifier...")
        pipeline.fit(X_train[:5000, :], y_train[:5000])

        print("\t\tPredicting...")
        y_true, y_pred = y_test[:1000], pipeline.predict(X_test[:1000])

        accuracy, f_scores = calc_results(y_true, y_pred, verbose=1)

        acc_avg += accuracy
        fsc_avg += f_scores

    acc_avg /= len(participants)
    fsc_avg /= len(participants)

    if verbose > 0:
        print("\tCV Results:")
        print("\t\tAvg Accuracy:", acc_avg)
        print("\t\tAvg F Score:", fsc_avg)

    return acc_avg, fsc_avg


def calc_results(y_true, y_pred, verbose=0):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f_scores, support = precision_recall_fscore_support(y_true, y_pred, average="macro")

    if verbose > 0:
        print("\t\t\tAccuracy:", acc)
        print("\t\t\tF Score (macro):", f_scores)
        # print("\t\t\tConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print("\t\t\tConfusion Matrix")
        [print("\t\t\t[{:8d} {:8d} {:8d}]".format(row[0], row[1], row[2])) for row in confusion_matrix(y_true, y_pred)]
        print()
        print(classification_report(y_true, y_pred))


    return acc, f_scores


def main():

    participants = ("p1", "p3", "p4", "p6")
    # feature_types = ("hog", "hog2")
    feature_types = ("hog", )

    # Setup Pipeline for Classification
    # steps = [('Resampling', SMOTE(kind="borderline2")), ("SVC", SVC(C=1, gamma=.001, kernel='rbf', probability=False))]
    steps = [('Resampling', ClusterCentroids(n_jobs=2)), ("SVC", SVC(C=1, gamma=.001, kernel='rbf', probability=False))]
    pipeline = Pipeline(steps)

    M = 5
    radius = .15

    verbose = 2
    rem_static = 2

    results = {}

    for feature_type in feature_types:

        print()
        print("###### Testing Feature Type: {} ######".format(feature_type))
        print("######################################")

        data = load_data("all_participants", M, radius, feature_type, "train")

        X_comb = np.vstack((data["X_lh"], data["X_rh"]))
        y_comb = np.hstack((data["y_lh"], data["y_rh"]))
        f_comb = np.hstack((data["filenames"], data["filenames"]))

        acc_avg, fsc_avg = cross_validation(pipeline, X_comb, y_comb, f_comb, participants, rem_static=rem_static, verbose=verbose)

        results[feature_type + "_acc"] = acc_avg
        results[feature_type + "_fsc"] = fsc_avg

    print("All Results")
    print("\t", results)
    # pickle.dump(results, open("posture_eval_results.pkl", "wb"))


if __name__ == '__main__':

    main()