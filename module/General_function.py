import numpy as np

def image_vector(image):
    """
    图片的红绿蓝三色亮度所组成的矩阵（3，x，y）转换为（3*y*x，1）的矩阵
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    vector=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return vector

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    x_exp = np.exp(x) # (n,m)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) # (n,1)

    # Compute softmax(x) by dividing x_exp by x_sum automatically use numpy broadcasting.
    s = x_exp / x_sum  # (n,m) 广播的作用. It should

    ### END CODE HERE ###

    return s

# -*- coding: utf-8 -*-

