import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)

    hypothesis = sigmoid(np.dot(X, theta))

    J = - np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)) / m
    grad = np.dot(X.T, (hypothesis - y)) / m

    return J, grad