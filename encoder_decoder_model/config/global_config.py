#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-31 上午11:21
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
設置全局變量
"""
from easydict import EasyDict as edict
# easydict的作用：可以使得以属性的方式去访问字典的值！在深度学习中往往利用easydict建立一个全局的变量

__C = edict()
# Consumers can get config by:" from config import cfg"

cfg = __C

# Train options # 创建一个字典，key是Train,值是{}
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 582690#300100#624010
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 1000
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.000005
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.85
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 1
# 設定T(non-local一次參考幾張圖的關係)
__C.TRAIN.T = 1

# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 1
# Set the learning rate decay steps

__C.TRAIN.LR_DECAY_STEPS = 510000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 2
# Set the image height
__C.TRAIN.IMG_HEIGHT = 256
# Set the image width
__C.TRAIN.IMG_WIDTH = 512
# Set GPU number
__C.TRAIN.GPU_NUM = 1
# Set CPU thread number
__C.TRAIN.CPU_NUM = 1

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
# Set the test batch size
__C.TEST.BATCH_SIZE = 32
