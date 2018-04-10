# -*- coding: utf-8 -*-
import numpy as np

class Deeplearn:
    """
    深度学习代码
    """
    
    def __init__(self,X_num, Unit,W = [],B = [],F = []):
        """
        初始化参数w,b,f
        Argument:
            W:各层次w参数取值
            B:各层次b参数取值
            F:各层次激活函数
            X_num:输入值个数
            Unit:表示初始模型各层隐藏单元个数
                eg：（4，4，2，1）表示共4层，1-4层隐藏单元个数分别为4，4，2，1
        """
        if W == [] :
            for n in range(len(Unit)):
                if n == 0:
                   W.append(np.random.random((Unit[n],X_num)))
                else:
                    W.append(np.random.random((Unit[n],Unit[n-1]))) 
        else:
            for n in range(len(W)):
                try:
                if n == 0:
                    assert W[n].shape == (Unit[n],X_num)
                else:
                    assert W[n].shape == (Unit[n],Unit[n-1])
                except:

        
        
    def Check_parameter(W,B,X,F,Unit):
        """
        
        
        """
        a=2
        
        
        
    def Forward_propagation(W,B,X,F,Unit):
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
        w=3
        
