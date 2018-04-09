# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from module import Activation_function
from module import Dlear_function


def draw():
    x=np.arange(-7,7,0.5)
    print(x)
    print(function.sigmoid(x))
    plt.plot(x,function.sigmoid(x),'r-')
    plt.ylabel('test y numbers')
    plt.xlabel("test x numbers")
    plt.show()
    plt.plot(x,function.sigmoid_derivative(x),'r-')
    plt.ylabel('test y numbers')
    plt.xlabel("test x numbers")
    plt.show()


def main():
    Dlear_function.Forward_propagation(1,2,0,Activation_function.sigmoid)
    
main()
