import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)

    hypothesis = sigmoid(np.dot(X, theta))

    reg_theta = theta[1:]

    J = - np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)) / m + (lmd / (2 * m)) * np.sum(reg_theta * reg_theta)

    normal_grad = (np.dot(X.T, hypothesis - y) / m).flatten()

    grad[0] = normal_grad[0]
    grad[1:] = normal_grad[1:] + reg_theta * (lmd / m)

    return J, grad