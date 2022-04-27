import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]
    p = np.zeros(m)

    p = sigmoid(np.dot(X, theta))
    pos = np.where(p >= 0.5)
    neg = np.where(p <= 0.5)

    p[pos] = 1
    p[neg] = 0

    return p