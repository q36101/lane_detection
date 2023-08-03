# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:42:48 2019

@author: mediacore
"""
import numpy as np
from PIL import Image
import cv2
import os

path='C:/Users/HLY/Downloads/culane/val_instance/'
newpath='C:/Users/HLY/Downloads/culane/val_instance1/'
# newpath='C:/Users/HLY/lane_detection/data/BOSCH/images-2014-12-18-14-17-05_binary/'

def turnto24(path):
   files = os.listdir(path)
   files = np.sort(files)
   for f in files:
       imgpath = path + f
       img = cv2.imread(imgpath)
    #    img = cv2.resize(img, (1276, 717))
       img = cv2.resize(img, (1640, 590))
       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
       image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       file_name, file_extend = os.path.splitext(f)
       dst = os.path.join(os.path.abspath(newpath), file_name + '.png')
       cv2.imwrite(dst, image_gray)
       #img.save(dst)

turnto24(path)