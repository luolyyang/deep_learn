# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:01:43 2018

@author: luoly
"""

import numpy as np
import time

#np.arange(2，10，1) 创建以2开始，步长为1，直到10的数组 
a=np.arange(100000)
b=np.arange(100000)
tic=time.process_time()
dot = np.zeros(100000)
for i in range(len(a)):
    dot[i] = a[i]+b[i]
    
toc=time.process_time()
print(dot[100
          ])
print(tic,toc)
print("yun xing shi chang:"+str(1000*(toc-tic)))



tic=time.process_time()
dot = np.zeros(100000)
dot = a + b
toc=time.process_time()
print(dot[100])
print(tic,toc)
print("yun xing shi chang:"+str(1000*(toc-tic)))