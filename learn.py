# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from module import Activation_function


def draw():
    x=np.arange(-7,7,0.5)
    plt.plot(x,Activation_function.sigmoid_or_gradient(x,True),'r-')
    plt.ylabel('test y numbers')
    plt.xlabel("test x numbers")
    plt.show()
    plt.plot(x,Activation_function.sigmoid_or_gradient(x,False),'r-')
    plt.ylabel('test y numbers')
    plt.xlabel("test x numbers")
    plt.show()


def main():
    draw()
    
main()
