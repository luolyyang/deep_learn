#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:39:20 2018
一般使用方法
无源模型
1、初始化模型 init_module 以及参数 init_parameter
2、验证模型维度 check_parameter
3、向前或者反向传播
已有源模型
1、读取源模型参数  read_module_from_pickle
2、验证模型维度 check_parameter
3、向前或者反向传播
@author: roy
"""
import numpy as np
import pickle
import os
import time
import sys


class DeepLearn:
    """
    深度学习代码
    """
    
    def __init__(self, module_name, work_path = ".", learn_rate = 0.1):
        """
        定义类，如初始化模型需要使用函数init_module初始化模型以及函数init_parameter初始化参数
                如已存在。需加载模型，需执行函数read_pickle获取原有模型信息
        :param module_name: 模型名称
                work_path： 工作路径
                learn_rate： 学习率 默认为0.1
        """
        self.flag = 1
        self.X_num = 0
        self.Unit = []
        self.B = []    #参数B
        self.W = []      #参数W
        self.F = []      #参数F 激活函数
        self.Z = [None]     #每层Z的值
        self.A = []     #每层输出结果
        self.Y = []     #训练数据所需的输出结果值,，人为统计
        self.dW = [None]    #梯度W
        self.dB = [None]    #梯度B
        self.dA = [None]    #梯度A
        self.dZ = [None]    #梯度Z
        self.work_path = work_path
        self.learn_rate = learn_rate
        self.module_name = module_name

    def storge_module_to_pickle(self):
        pickle_file = self.work_path + "/parameter_pickle/" + self.module_name + "_parameter.pickle.file"
        if os.path.exists(pickle_file):
            os.rename(pickle_file,pickle_file + "_" + time.strftime('%Y%m%d%H%M%S',time.localtime()))
        file = open(pickle_file,'wb')
        pickle.dump({'W':self.W,'B':self.B,'F':self.F,'Unit':self.Unit,'X_num':self.X_num,'learn_rate':self.learn_rate},file)
        file.close()

    def read_module_from_pickle(self):
        """
        读取配置文件中存储的参数W,B,F
        """
        pickle_file = self.work_path + "/parameter_pickle/" + self.module_name + "_parameter.pickle.file"
        if os.path.exists(pickle_file) == False:
            print(pickle_file + "文件不存在，请确认pickle文件状态或者初始化模型")
        else:
            file = open(pickle_file, 'rb')
            data = pickle.load(file)
            self.W = data['W']
            self.B = data['B']
            self.F = data['F']
            self.X_num = data['X_num']
            self.Unit = data['Unit']
            self.learn_rate = data["learn_rate"]
            file.close()

    def init_module(self, x_num, unit, f ,w = [], b = []):
        """
        初始化模型，必传参数 x_num unit f,若w,b均已提供，请勿执行函数init_parameter初始化w，b，
        直接执行函数check_parameter检查维度
        :param x_num:  输入维度
        :param unit:  模型深度信息,表示初始模型各层隐藏单元个数
                eg：（4，4，2，1）表示共4层，1-4层隐藏单元个数分别为4，4，2，1
        :param f:  各层次使用的激活函数
        :param w:  默认值为空列表，W值
        :param b: 默认值为空列表 ，B值
        :return:
        """
        self.Unit = [None] + unit
        self.X_num = x_num
        self.W = [None] + w
        self.B = [None] + b
        self.F = [None] + f

    def init_parameter(self):
        """
        初始化w，b参数，生成满足矩阵维度的W,B列表
        :return:
        """
        if self.Unit == [None] or self.X_num == 0:
            print("请先使用init_module函数初始化模型: X_num，Unit, F")
            sys.exit()
        else:
            for l in range(1,len(self.Unit), 1):
                if l == 1:
                    self.W.append(np.random.random((self.Unit[l],self.X_num)))
                else:
                    self.W.append(np.random.random((self.Unit[l],self.Unit[l-1])))
                self.B.append(np.random.random((self.Unit[l],1)))

    def init_back_parameter(self):
        """
        初始化反向传播（梯度下降）所需参数 dw, db ,da ,dz
        :return:
        """
        if self.W == [None] or self.B == [None]:
            print("请先使用init_parameter函数初始化参数: w，b或者使用函数read_module_from_pickle读取模型信息！")
            sys.exit()
        else:
            for l in range(1, len(self.Unit), 1):
                self.dW.append(np.zeros_like(self.W[l]))
                self.dB.append(np.zeros_like(self.B[l]))
                self.dA.append(np.zeros_like(self.B[l]))
                self.dZ.append(np.zeros_like(self.B[l]))

    def check_parameter(self):
        """
        检查各参数维度是否满足合理性
        Returns:
            FLAG:验证成功与否标识位，False 失败，True 成功
        """
        for l in range(1, len(self.Unit), 1):
            try:
                if l == 1:
                    assert self.W[l].shape == (self.Unit[l],self.X_num)
                else:
                    assert self.W[l].shape == (self.Unit[l],self.Unit[l-1])
            except:
                print("第" + str(l) + "层w参数维度不和，请使用init_parameter初始化参数或者init_module初始化模型！")
                self.flag = 0
        for l in range(1, len(self.Unit), 1):
            try:
                assert self.B[l].shape == (self.Unit[l], 1)
            except:
                print("第" + str(l) + "层b参数维度不和，请使用init_parameter初始化参数或者init_module初始化模型！")
                self.flag = 0
        try:
            assert len(self.F) == len(self.Unit)
        except:
            print("F激活函数维度不和，请使用init_module初始化模型！")
            print("激活函数len(F)=" + str(len(self.F)) + "模型len(Unit)=" + str(len(self.Unit)))
            self.flag = 0
        if self.flag == 0:
            return False
        else:
            return True

    def forward_propagation(self, x):
        """
        前向传播
        Argument:
            X输入
            eg x:
                [[in1，in1, ...]
                 [in2, in2, ...]
                 [in3, in3, ...]]
                 数据1 数据2
        """
        #Z=W*X+B，Y=g(Z)
        self.A.append(x)
        for l in range(1, len(self.Unit), 1):
            self.Z.append(np.dot(self.W[l],self.A[l-1]) + self.B[l])
            self.A.append(self.F[l](self.Z[l], True))

    def back_propagation(self, y):
        """
        计算反反向传播（梯度）
        :param Y:
            输入训练数据输出值Y
        :return:
            无
        """
        self.init_back_parameter()
        m = y.shape[1]
        for l in range(len(self.Unit)-1, 0, -1):
            if l == len(self.Unit) - 1:
                self.dA[l] = -(y / self.A[l]) + (1 - y) / (1 - self.A[l])
            else:
                self.dA[l] = np.dot(self.W[l+1].T,self.dZ[l+1])
            self.dZ[l] = self.dA[l]*self.F[l](self.Z[l],False)
            self.dW[l] = 1/m*np.dot(self.dZ[l] , self.A[l-1].T)
            self.dB[l] = 1/m*np.sum(self.dZ[l],axis=1,keepdims=True)

    def update_parameter(self,learn_rate = None):
        """
        更新w和b的参数,学习率可更新，更新后自动保存至pickle文件中
        :param learn_rate: 学习率 在类定义时已指定，若要使用新的学习率请指定，初始值 None
        :return: 无
        """
        if learn_rate == None:
            pass
        else:
            self.learn_rate = learn_rate
        for l in range(1, len(self.Unit),1):
            self.W[l] += self.dW[l]*self.learn_rate
            self.B[l] += self.dB[l]*self.learn_rate

    def cost_function(self,Y):
        print(len(self.Unit)-1)
        print
        cost = -(1.0 / Y.shape[1]) * np.sum(Y * np.log(self.A[len(self.Unit)-1]) + (1 - Y) * np.log(1 - self.A[len(self.Unit)-1]))
        return cost





