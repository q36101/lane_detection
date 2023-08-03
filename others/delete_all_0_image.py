# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:52:15 2020

@author: mediacore
"""

import shutil
import imghdr
import cv2
import os
import numpy as np
import glob
import os.path as ops

# image_path_1 = 'C:/Users/HLY/Downloads/bdd100k_lane_labels_trainval/bdd100k/labels/lane/colormaps/train_c/'
# image_path_2 = 'C:/Users/HLY/Downloads/bdd100k_lane_labels_trainval/bdd100k/labels/lanecolormaps/train2/'
path='C:/Users/HLY/Downloads/bdd100k_lane_labels_trainval/bdd100k/labels/lane/colormaps/'
image_path_1 = ops.join(path, 'delet.txt')
# file = os.listdir(image_path_1)
# files = os.listdir(image_path_2)

#for i in files:
#    path = image_path_2 + i
#    imgtype = imghdr.what(path)     #查看資料夾內的檔案格式(副檔名)
#    if(imgtype != 'png'):
#        print(path, '!!!!!!')

# for i in file:                   #對照兩個資料夾 檢查file(path_1)裡的資料是否在path_2也有 若path_2沒有就把path_1裡的那個資料刪除
#     path_1 = image_path_1 + i
#     path_2 = image_path_2 + i
#     if(os.path.isfile(path_2)):
#         print(i)
#     else:
#         print(path_1, 'NNN')
#         os.remove(path_1)

with open(image_path_1 ,'r') as f:
    # data=f.read().split(',')
    l1 = f.readlines()
    
    print(l1)
    os.remove(l1)
    

#     if os.path.isfile(i) :
#         os.remove(i)
#     else:
#         i+1   
    
# for i in data :
#     print(i)

#     if os.path.isfile(i) :
#         os.remove(i)
#     else:
#         i+1

# for i in file:
#    path = image_path_1 + i
   
#    image = cv2.imread(path, cv2.IMREAD_COLOR)
#    if(np.sum(image) ==0 ):
#        print(path)        
#        os.remove(path)
# path='C:/Users/HLY/Downloads/bdd100k_lane_labels_trainval/bdd100k/labels/lane/colormaps/train3/'

# path=glob.glob(os.path.join(path,'*.png'))

# for file in path:
#     image = cv2.imread(file, 0)
#     if cv2.countNonZero(image)==255:
#         print(file)  
#         os.remove(file)    
    
       
        
        
    
