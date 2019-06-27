import sys
sys.path.append("./")

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.decomposition import PCA

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, ClusterCentroids, RepeatedEditedNearestNeighbours

from vpt.common import *

def load_data(testing_p, M, radius, feature_type, data_type):
    base = "data/posture/extracted/"
    data_path = os.path.join(base, "{}_M{}_rad{:0.2f}_{}_".format(testing_p, M, radius, feature_type))
    data = np.load(data_path + data_type + "-new_data_combined.npz")
    return data


def split_static(X, y, f):
    # remove p#s data
    r = re.compile('p[\d]s')
    vmatch = np.vectorize(lambda x: bool(r.search(x)))
    rem_static = vmatch(f)

    X_nos, y_nos, f_nos = X[~rem_static], y[~rem_static], f[~rem_static]
    X_s, y_s, f_s = X[rem_static], y[rem_static], f[rem_static]

    return X_nos, y_nos, f_nos, X_s, y_s, f_s


def split_participant(X, y, f, participant):
    r = re.compile(participant)
    vmatch = np.vectorize(lambda x: bool(r.search(x)))
    val_p = vmatch(f)

    X_p = X[val_p]
    y_p = y[val_p]
    f_p = f[val_p]

    return X_p, y_p, f_p


def cross_validate_windows(pipeline, X, y, f, size, n_splits, verbose=0):

    groups = np.zeros_like(f, dtype=int)
    for i in range(len(f) // size):
        groups[i * size:i * size + size] = i
    groups[len(groups) % size * -1:] = i + 1

    cv = GroupKFold(n_splits=n_splits)
    print("### Window Based - size: {}".format(size))
    return cross_validate(pipeline, cv, X, y, groups, verbose)


def cross_validate_exercises(pipeline, X, y, f, exercise_names, verbose=0):

    groups = np.ones_like(f, dtype=int)
    for i, ex in enumerate(exercise_names):
        r = re.compile("p[\d]{}".format(ex))
        vmatch = np.vectorize(lambda x: bool(r.search(x)))
        ex_idxs = vmatch(f)

        p_num = i
        groups[ex_idxs] = p_num

    print(np.unique(groups, return_counts=True))

    cv = LeaveOneGroupOut()
    print("### Exercise Based")
    return cross_validate(pipeline, cv, X, y, groups, verbose)


def static_cv_exercises(pipeline, X_train, y_train, f_train, X_test, y_test, f_test, exercises, verbose=0):

    results = {}

    for ex in exercises:
        print("#### EX: {} ####".format(ex))

        r = re.compile('p[\d]{}'.format(ex))
        vmatch = np.vectorize(lambda x: bool(r.search(x)))
        ex_match = vmatch(f_test)

        X_test_ex, y_test_ex, f_test_ex = X_test[ex_match], y_test[ex_match], f_test[ex_match]

        if verbose > 0:
            print("Train Data - X:", X_train.shape, "y:", y_train.shape)
            print("Test Data - X:", X_test_ex.shape, "y:", y_test_ex.shape)

        print("Training...")
        pipeline.fit(X_train, y_train)
        print("Predicting...")
        y_true, y_pred = y_test_ex, pipeline.predict(X_test_ex)

        results["ex{}_accuracy".format(ex)] = accuracy_score(y_true, y_pred)
        results["ex{}_conf_mat".format(ex)] = confusion_matrix(y_true, y_pred)
        results["ex{}_f_score".format(ex)] = f1_score(y_true, y_pred, average="weighted")
        results["ex{}_report".format(ex)] = classification_report(y_true, y_pred)

        print("Results:")
        print("Accuracy:", results["ex{}_accuracy".format(ex)])
        print("F1:", results["ex{}_f_score".format(ex)])
        print("Confusion Matrix:\n", results["ex{}_conf_mat".format(ex)])
        print(results["ex{}_report".format(ex)])
        print()

    return results


def cross_validate(pipeline, cv, X, y, groups, verbose=0):


    param_grid = [
        {'SVC__C': [0.01, 0.11, 10, 100],
         'SVC__gamma': [.001, .01, .1, 1, 10],
         'SVC__kernel': ['rbf']},
        {'SVC_C': [0.01, 0.11, 10, 100],
         'SVC_kernel': ['linear']}
    ]

    scoring = ['f1_macro', 'f1_micro', 'accuracy']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("## Tuning hyper-parameters ##")
        print()

        clf_comb = GridSearchCV(pipeline, param_grid, cv=cv.split(X, y, groups=groups),
                                scoring=scoring, n_jobs=-1, verbose=1, refit="accuracy")
        clf_comb.fit(X, y)

        print("Best Combined Parameters set found on data set:")
        print()
        print(clf_comb.best_params_)
        print()
        print("Grid scores on data set:")
        print()
        means_acc = clf_comb.cv_results_['mean_test_accuracy']
        stds_acc = clf_comb.cv_results_['std_test_accuracy']
        means_recall = clf_comb.cv_results_['mean_test_recall_macro']
        stds_recall = clf_comb.cv_results_['std_test_recall_macro']
        for mean_acc, std_acc, mean_recall, std_recall, params in zip(means_acc, stds_acc, means_recall,
                                                                      stds_recall, clf_comb.cv_results_['params']):
            print("Recall: %0.3f (+/-%0.3f) - Accuracy: %0.3f (+/-%0.3f) for\n %r" % (
            mean_recall, std_recall, mean_acc, std_acc, params))
            print()
        print()


    return clf_comb.cv_results_


def main(M, radius, pipeline, feature_type, participants, exp_num, exp_name, cv, *args, rem_static=True, verbose=0):

    #### Loading all data
    data = load_data("all_participants", M, radius, feature_type, "train")

    X_lh = data['X_lh']
    y_lh = data['y_lh']
    f_lh = data['filenames']
    X_rh = data['X_rh']
    y_rh = data['y_rh']
    f_rh = data['filenames']

    if rem_static:
        X_lh, y_lh, f_lh, _, _, _ = split_static(X_lh, y_lh, f_lh)
        X_rh, y_rh, f_rh, _, _, _ = split_static(X_rh, y_rh, f_rh)

    X = np.vstack((X_lh, X_rh))
    y = np.hstack((y_lh, y_rh))
    f = np.hstack((f_lh, f_rh))

    print("### Data Loaded: ###")
    print("X LH", X_lh.shape, "y LH", y_lh.shape, "Files", f_lh.shape)
    print("X RH", X_rh.shape, "y RH", y_rh.shape, "Files", f_rh.shape)
    print("X Comb:", X.shape, "y Comb", y.shape, "Files", f.shape)
    print()

    results = {}
    for p in participants:

        X_p, y_p, f_p = split_participant(X, y, f, p)

        print("##### Parameter Tuning for {} #####".format(p))
        if verbose > 1:
            print("Data Counts - X:", X_p.shape, " - y:",  y_p.shape)
        results[p] = cv(pipeline, X_p, y_p, f_p, *args, verbose=verbose)

    pickle.dump(results, open("cmj_exp-{}_results_{}.pkl".format(exp_name, exp_num), "wb"))


if __name__ == '__main__':

    from vpt.classification.hierarchical_clf import HierarchicalClassifier

    participants = ["p3", "p1", "p4", "p6"]

    M = 5
    radius = .15

    rem_static = True
    verbose = 2

    cv = cross_validate_windows
    window_size = 30
    k_folds = 3

    feature = "honv"
    cell_size = (8,8)
    block_size = (1,1)

    exp_num = 0
    exp_name = "defense"

    # feature_type = "f_{}-c_{}-b_{}".format(feature, cell_size[0], block_size[0])
    feature_type = "test"
    # cv = cross_validate_exercises
    # exercises = ["a", "b", "c", "d", "e"]


    steps = [("SVC", SVC(C=10, kernel="linear", decision_function_shape='ovr', probability=False))]

    # clfs = [Pipeline(steps1), Pipeline(steps2)]
    # pos = (0, 1)
    # neg = ((1,2), (2,))
    #
    # clf = HierarchicalClassifier(clfs, pos, neg)

    clf = Pipeline(steps)

    main(M, radius, clf, feature_type, participants, exp_num, exp_name, cv, window_size, k_folds, rem_static=rem_static, verbose=verbose)
    # main(M, radius, clf, feature_type, participants, exp_num, cv, exercises, rem_static=rem_static, verbose=verbose)