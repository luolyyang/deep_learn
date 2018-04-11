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
        self.flag = 1
        self.X_num = X_num
        self.Unit = Unit
        self.B = B
        self.W = W
        self.F = F
        self.Z = []
        self.A = []
        
        
    def Init_parameter(self):
        for n in range(len(self.Unit)):
            if n == 0:
                self.W.append(np.random.random((self.Unit[n],self.X_num)))
            else:
                self.W.append(np.random.random((self.Unit[n],self.Unit[n-1])))
            self.B.append(np.random.random((self.Unit[n],1)))

            
            
    def Check_parameter(self):
        """
        检查各参数维度是否合格
        Returns:
            FLAG:验证成功与否标识位，False 失败，True 成功
        """
        for n in range(len(self.W)):
            try:
                if n == 0:
                    assert self.W[n].shape == (self.Unit[n],self.X_num)
                else:
                    assert self.W[n].shape == (self.Unit[n],self.Unit[n-1])
            except:
                print("第"+ str(n) +"层w参数维度不和，请确认！")
                self.flag = 0
        for n in range(len(self.Unit)):
            try:
                assert self.B[n].shape == (self.Unit[n],1)
            except:
                print("第"+ str(n) +"层b参数维度不和，请确认！")
                self.flag = 0
            try:
                assert self.F[n].shape == (self.Unit[n],1)
            except:
                print("第"+ str(n) +"层f激活函数维度不和，请确认！")
                self.flag = 0
        if self.flag == 0:
            return False
        else:
            return True
        
        
        
        
    def Forward_propagation(self,X):
        """
        Argument:
            X输入
        """
        #Z=W*X+B，Y=g(Z)
        self.A.append(X)
        for n in range(len(self.Unit)):
            self.Z.append(np.dot(self.W[n],self.A[n])+self.B[n])
            self.A.append(self.F[n](self.Z[n]))
        
    
    def Back_propagation():
        w=3
        
