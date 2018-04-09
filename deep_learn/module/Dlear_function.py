# -*- coding: utf-8 -*-
import numpy as np


def Forward_propagation(W,B,X,F):
    """
    正向传播函数
    Argument:
    W:当前函数参数w所组成的矩阵
    B:当前常数参量b所组成的矩阵
    X:输入
    F:激活函数
    Returns:
    Y:当前参数所计算输出结果
    """
    #Z=W*X+B，Y=g(Z)
    Z=np.dot(W,X)+B
    Y=F(Z)
    return Z,Y

def Back_propagation():
    
    
