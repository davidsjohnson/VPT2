from vpt.common import *

class GenericClassifier:

    def __init__(self):
        self._label = None

    def fit(self, X, y):
        self._label = np.max(y)

    def predict(self, X):
        return np.ones(len(X)) * self._label




class HierarchicalClassifier:

    def __init__(self, clfs, pos_labels, neg_labels):

        self._clfs = clfs
        self._pos_labels = pos_labels
        self._neg_labels = neg_labels


    def fit(self, X, y):

        for i, (clf, pos, neg) in enumerate(zip(self._clfs, self._pos_labels, self._neg_labels)):

            y_temp = np.ones_like(y) * -1

            y_temp[np.where(y == pos)] = 0

            for l in neg:
                y_temp[np.where(y == l)] = 1

            print("Fitting with:")
            print("X:", X[y_temp != -1].shape, "y:", y_temp[y_temp != -1].shape)
            print("OG Counts", np.unique(y[y_temp != -1], return_counts=True))
            print("Binary Counts", np.unique(y_temp[y_temp != -1], return_counts=True))

            if len(np.unique(y_temp[y_temp != -1])) <= 1:
                print("Using Generic Classifier...")
                clf = GenericClassifier()
                self._clfs[i] = clf
                clf.fit(X[y_temp != -1], y_temp[y_temp != -1])
            else:
                clf.fit(X[y_temp != -1], y_temp[y_temp != -1])


    def predict(self, X):

        y_final = np.ones(len(X), dtype=int) * -1

        for i, (clf, pos, neg) in enumerate(zip(self._clfs, self._pos_labels, self._neg_labels)):

            y_temp = np.ones_like(y_final) * -1

            print("Predicting with:")
            print("X:", X[y_final == -1].shape)

            y_temp[y_final == -1] = clf.predict(X[y_final == -1])

            print("counts", np.unique(y_temp[y_final == -1], return_counts=True))

            y_final[y_temp == 0] = pos

            if i == (len(self._clfs) - 1):
                y_final[y_temp == 1] = neg[0]

        return y_final


def unit_test_validate():

    X, y = load_iris(return_X_y=True)

    clfs = [SVC(random_state=0), SVC(random_state=0)]
    pos_labels = [0, 1]
    neg_labels = [(1, 2), (2,)]

    clf = HierarchicalClassifier(clfs, pos_labels, neg_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    clf.fit(X_train, y_train)

    y_true, y_pred = y_test, clf.predict(X_test)

    print("Hierarchical Clf")
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print()

    print("Std SVC")
    svc = SVC(random_state=0)
    svc.fit(X_train, y_train)
    y_true, y_pred = y_test, svc.predict(X_test)
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print()


def unit_test_pipeline():

    X, y = load_iris(return_X_y=True)

    steps = [("Smote", SMOTE(random_state=1)), ("Clf", SVC(random_state=0))]
    steps2 = [("Smote", SMOTE(random_state=1)), ("Clf", SVC(random_state=0))]

    clfs = [Pipeline(steps), Pipeline(steps2)]
    pos_labels = [0, 1]
    neg_labels = [(1, 2), (2,)]

    clf = HierarchicalClassifier(clfs, pos_labels, neg_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    clf.fit(X_train, y_train)

    y_true, y_pred = y_test, clf.predict(X_test)

    print("Hierarchical Clf w/ Pipeline")
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print()

    print("Std Pipeline")
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    y_true, y_pred = y_test, pipeline.predict(X_test)
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print()


if __name__ == '__main__':

    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    unit_test_validate()
    unit_test_pipeline()



