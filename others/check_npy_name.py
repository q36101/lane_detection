# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:36:45 2020

@author: mediacore
"""


import numpy as np
 
data_dict = np.load('C:/Users/mediacore/lane_detection/data/mobilenet_v3_large.npy', allow_pickle=True, encoding='latin1').item()
keys = sorted(data_dict.keys())
for key in data_dict:

    print('\n')
    print(key)

