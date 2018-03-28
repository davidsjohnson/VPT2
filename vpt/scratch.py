from sklearn.svm import SVC

from vpt.common import *
from vpt.individual_eval import *

M = 5
radius = .15
feature_type = "hog"
rem_static = True

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