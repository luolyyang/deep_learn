# -*- coding: utf-8 -*-
import numpy as np


def sigmoid_or_gradient(x,flag):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size
    flag -- A flag with return derivative of sigmoid
    Return:
    if flag is True
    return s -- sigmoid(x)
    if flag is False
    return s -- derivative values of sigmoid(x)
    """
    s = 1.0 / (1 + 1 / np.exp(x))
    if flag:
        return s
    else:
        ds = s * (1 - s)
        return ds

