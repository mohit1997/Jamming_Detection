import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

x = np.load('input_symbols.npy')
x_ = np.load('predictions.npy')

print(x.shape)

X = x.reshape(-1)
X_ = x_.reshape(-1)

X_[X_ < 0.5] = 0.0
X_[X_ >= 0.5] = 1.0

print(np.mean(X==X_))

# X = X.astype(np.int32)
print(x.dtype)

print("Mututal Information for the NN is ", calc_MI(X, X_, 20)/np.log(2))
print("Mututal Information for the NN is ", mutual_info_classif(x, x_)/np.log(2))