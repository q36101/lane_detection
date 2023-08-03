# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:25:58 2019

@author: mediacore
"""
"""
實現一個基於MoblieNet_v2的特徵編碼類
"""
from collections import OrderedDict
import math
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from tensorflow.keras.layers import (Layer,Input,LayerNormalization,
                                    Dense,Dropout,Conv2D,)
from tensorflow import keras
from tensorflow.keras import layers
# from custom_function import (drop_path)
# from custom_function import window_partition, window_reverse
# from custom_layer import SwinTransformerBlockLayer ,PatchEmbeddingLayer,PatchMergingLayer
from encoder_decoder_model import cnn_basenet_swin
# import cnn_basenet_swin
from einops import rearrange
# import cnn_basenet
from config import global_config


CFG = global_config.cfg
weight_decay=1e-5

def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf
# （1）Embedding 层
# inputs代表输入图像，shape为224*224*3
# out_channel代表该模块的输出通道数，即第一个卷积层输出通道数=768
# patch_size代表卷积核在图像上每16*16个区域卷积得出一个值
# --------------------------------------------- #
def patch_embed(inputs, out_channel, patch_h,patch_w):
    
    # 获得输入图像的shape=[b,224,224,3]
    b, h, w, c = inputs.shape
    print('b, h, w, c =',b, h, w, c)
    # 获得划分后每张图像的size=(14,14)
    grid_h, grid_w = h//patch_h, w//patch_w
    print('grid_h, grid_w =',grid_h, grid_w)
    # 计算图像宽高共有多少个像素点 n = h*w
    num_patches = grid_h * grid_w#
# 
    # x = layers.Conv2D(filters=out_channel*patch_h*patch_w, kernel_size=(patch_h,patch_w), strides=(patch_h,patch_w) ,padding='same')(inputs)
 
    # # 维度调整 [b,h,w,c]==>[b,n,c]
    # # [b,14,14,768]==>[b,196,768]
    # print('xxxxxxxxxxxxxxx',num_patches, out_channel, patch_h, patch_w)
    print('xxxxxxxxxxxxxxx',inputs)
    x = rearrange(inputs,'b (h p1) (w p2) c -> b (h w)  (p1 p2 c)',p1=grid_h,p2=grid_w)
    print('xxxxxxxxxxxxxxx',x)   
    return x
def class_pos_add(inputs):

    # 获得输入特征图的shape=[b,196,768]
    b, num_patches, channel = inputs.shape
    num_patches=int(num_patches)
    channel=int(channel)
    # 直接通过classtoken来判断类别，classtoken能够学到其他token中的分类相关的信息
    cls_token = layers.Layer().add_weight(name='classtoken', shape=[1,1,channel], dtype=tf.float32,
                                        initializer=keras.initializers.Zeros(), trainable=True)  
    # 可学习的位置变量 [1,197,768], 初始化为0，trainable=True代表可以通过反向传播更新权重
    pos_embed = layers.Layer().add_weight(name='posembed', shape=[1,num_patches+1,channel], dtype=tf.float32,
                                        initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True)
    cls_token = tf.broadcast_to(cls_token, shape=[b, 1, channel])
    # 在num_patches维度上堆叠，注意要把cls_token放前面
    x = layers.concatenate([cls_token, inputs], axis=1)
    # 将位置信息叠加上去
    x = tf.add(x, pos_embed)

    return x  # [b,197,768]
# --------------------------------------------- #
# （3）多头自注意力模块
# inputs： 代表编码后的特征图
# num_heads: 代表多头注意力中heads个数
# qkv_bias: 计算qkv是否使用偏置
# atten_drop_rate, proj_drop_rate：代表两个全连接层后面的dropout层
# --------------------------------------------- #
def attention(inputs, num_heads, qkv_bias=False, atten_drop_rate=0., proj_drop_rate=0.):

    # 获取输入特征图的shape
    b, num_patches, channel = inputs.shape#(2,33,1280)
    print('channelchannel',channel)
    # 计算head的通道数
    head_channel = channel // num_heads#16
    print('head_channel = channel // num_heads',channel , num_heads)
    
    # 公式的分母，根号d
    head_channel=int(head_channel)#16
    scale = head_channel ** 0.5 #4
    print('head',head_channel)

    # 经过一个全连接层计算qkv
    qkv = layers.Dense(channel*3, use_bias=qkv_bias)(inputs)
    qkv = tf.reshape(qkv, shape=[b, num_patches, 3, num_heads, channel//num_heads])
    qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
    # 获取q、k、v的值==>[b,num_heads,197,c//num_heads]
    q, k, v = qkv[0], qkv[1], qkv[2]
    # 矩阵乘法, q 乘 k 的转置，除以缩放因子。矩阵相乘计算最后两个维度
    # [b,num_heads,197,c//num_heads] * [b,num_heads,c//num_heads,197] ==> [b,num_heads,197,197]
    atten = tf.matmul(a=q, b=k, transpose_b=True) / scale #
    # 对每张特征图进行softmax函数
    atten = tf.nn.softmax(atten, axis=-1)
    # 经过dropout层
    atten = layers.Dropout(rate=atten_drop_rate)(atten)
    # 再进行矩阵相乘==>[b,num_heads,197,c//num_heads]
    atten = tf.matmul(a=atten, b=v)

    # 维度重排==>[b,197,num_heads,c//num_heads]
    x = tf.transpose(atten, perm=[0, 2, 1, 3])
    # 维度调整==>[b,197,c]==[b,197,768]
    x = tf.reshape(x, shape=[b, num_patches, channel])

    # 调整之后再经过一个全连接层提取特征==>[b,197,768]
    x = layers.Dense(channel)(x)
    # 经过dropout
    x = layers.Dropout(rate=proj_drop_rate)(x)

    return x
# ------------------------------------------------------ #
# （4）MLP block
# inputs代表输入特征图；mlp_ratio代表第一个全连接层上升通道倍数；
# drop_rate代表杀死神经元概率
# ------------------------------------------------------ #
def mlp_block(inputs, mlp_ratio=4.0, drop_rate=0.):

    # 获取输入图像的shape=[b,197,768]
    b, num_patches, channel = inputs.shape

    # 第一个全连接上升通道数==>[b,197,768*4]
    x = layers.Dense(int(channel*mlp_ratio))(inputs)
    # GeLU激活函数
    x = gelu(x)
    # original is "x = layers.Activation('gelu')(x)"but tensorflow has no GELU activation function,so I added gelu function.
    # dropout层
    x = layers.Dropout(rate=drop_rate)(x)

    # 第二个全连接层恢复通道数==>[b,197,768]
    x = layers.Dense(channel)(x)
    # dropout层
    x = layers.Dropout(rate=drop_rate)(x)

    return x
# ------------------------------------------------------ #
# （5）单个特征提取模块
# num_heads：代表自注意力的heads个数
# epsilon：小浮点数添加到方差中以避免除以零
# drop_rate：自注意力模块之后的dropout概率
# ------------------------------------------------------ #
def encoder_block(inputs, num_heads, epsilon=1e-6, drop_rate=0.):

    # LayerNormalization
    x = layers.LayerNormalization(epsilon=epsilon)(inputs)
    # 自注意力模块
    x = attention(x, num_heads=num_heads)
    # dropout层
    x = layers.Dropout(rate=drop_rate)(x)
    # 残差连接输入和输出
    # x1 = x + inputs
    x1 = layers.add([x, inputs]) 
    # LayerNormalization
    x = layers.LayerNormalization(epsilon=epsilon)(x1)
    # MLP模块
    x = mlp_block(x)
    # dropout层
    x = layers.Dropout(rate=drop_rate)(x)
    # 残差连接
    # x2 = x + x1
    x2 = layers.add([x, x1])
   
    return x2  
# ------------------------------------------------------ #
def transformer_block(x, num_heads):

    # 重复堆叠12次
    for _ in range(1):
        # 本次的特征提取块的输出是下一次的输入
        x = encoder_block(x, num_heads=num_heads)

    return x  # 返回特征提取12次后的特征图
# ---------------------------------------------------------- # 
def VIT(batch_shape, drop_rate=0., out_channel=192,num_heads=3, p=(6,16), t=(2,6,16,192) ,epsilon=1e-6,h=0,w=0):
 
    inputs=batch_shape
    # b,h,w,c = tf.shape(input)
    b, h, w, c = inputs.shape
    print('pppppppppppppppppppppp',p[0],p[1])
    x= patch_embed(inputs, out_channel,p[0],p[1])#1280)
    grid_h, grid_w = h//p[0], w//p[1]
    x = class_pos_add(x) 
    x = layers.Dropout(rate=drop_rate)(x)#(2, 3, 5120)
    print('11111111111',x)
    x = transformer_block(x, num_heads) #(2, 3, 5120)
    # LayerNormalization
    x = layers.LayerNormalization(epsilon=epsilon)(x)#(2, 3, 5120)
    # cls_ticks = x[:,0]#(2,5120)
    x = x[:,1:]

    print('rearrangerearrangerearrangerearrange',x)
    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh',b, h, w, c,grid_h,grid_w)
    x  = rearrange(x ,'b (h w)  (p1 p2 c) -> b (h p1) (w p2) c',h=int(h/grid_h),p1=int(grid_h),p2=int(grid_w))
    return x
class mobilenet_v3_large_Encoder(cnn_basenet_swin.CNNBaseModel):
    """
    實現了一個基於MoblieNet_v3的特徵編碼類
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(mobilenet_v3_large_Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)
    
    def separable_conv2d(self, input_tensor, out_dims, k_size, strides, padding, name, dilation_rate=(1, 1)):
        
        with tf.compat.v1.variable_scope(name):
            dsconv = tf.layers.separable_conv2d(input_tensor, filters=out_dims, kernel_size=k_size, strides=strides, padding=padding,
                                                dilation_rate=dilation_rate, name='dsconv')
            
            bn = self.layerbn(inputdata=dsconv, is_training=self._is_training, name='bn')

            relu = tf.nn.relu6(bn, name='relu')

            
        return relu
    
    def relu6(self, x, name='Relu6'):
        return tf.nn.relu6(x, name)
    
    def hard_swish(self, x,name='hard_swish'):
        with tf.name_scope(name):
            h_swish = x*tf.nn.relu6(x+3)/6
            return h_swish
    
                        #0.997
    def batch_norm(self, x, momentum=0.997, epsilon=1e-3, train=True, name='bn'):
        return tf.compat.v1.layers.batch_normalization(x,
                          momentum=momentum,
                          epsilon=epsilon,
                          scale=True,
                          center=True,
                          training=train,
                          name=name)
    
    
    def conv2d(self, input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.09, name='conv2d', bias=False):
        output_dim=int(output_dim)
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  regularizer=tf.keras.regularizers.l2(weight_decay),
                  initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
            if bias:
                biases = tf.compat.v1.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
    
            return conv
    
    
    def conv2d_block(self, input, out_dim, k, s, is_train, name, h_swish=False):
        with tf.compat.v1.name_scope(name), tf.compat.v1.variable_scope(name):
            net = self.conv2d(input, out_dim, k, k, s, s, name='conv2d')
            net = self.batch_norm(net, train=is_train, name='bn')
            if h_swish == True:
                net = self.hard_swish(net)
            else:
                net = self.relu6(net)
            return net
    
    
    def conv_1x1(self, input, output_dim, name, bias=False):
        with tf.compat.v1.name_scope(name):
            return self.conv2d(input, output_dim, 1,1,1,1, stddev=0.09, name=name, bias=bias)
    
    def pwise_block(self, input, output_dim, is_train, name, bias=False):
        with tf.name_scope(name), tf.compat.v1.variable_scope(name):
            out=self.conv_1x1(input, output_dim, bias=bias, name='pwb')
            out=self.batch_norm(out, train=is_train, name='bn')
            out=self.relu6(out)
            return out
    
    def dwise_conv(self, input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
                   padding='SAME', stddev=0.09, name='dwise_conv', bias=False):
        with tf.compat.v1.variable_scope(name):
            in_channel=input.get_shape().as_list()[-1]
            w = tf.compat.v1.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                            regularizer=tf.keras.regularizers.l2(weight_decay),
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.depthwise_conv2d(input, w, strides, padding,name=None,data_format=None)
            if bias:
                biases = tf.compat.v1.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
    
            return conv
    
    def Fully_connected(self, x, units, layer_name='fully_connected') :
        with tf.name_scope(layer_name) :
            return tf.compat.v1.layers.dense(inputs=x, use_bias=True, units=units)
    
    def global_avg(self, x,s=1):
        with tf.name_scope('global_avg'):
            net=tf.compat.v1.layers.average_pooling2d(x, x.get_shape()[1:-1], s)
            return net
    
    
    def hard_sigmoid(self, x,name='hard_sigmoid'):
        with tf.name_scope(name):
            h_sigmoid = tf.nn.relu6(x+3)/6
            return h_sigmoid
    
    def conv2d_hs(self, input, output_dim, is_train, name, bias=False,se=False):
        with tf.name_scope(name), tf.compat.v1.variable_scope(name):
            out=self.conv_1x1(input, output_dim, bias=bias, name='pwb')
            out=self.batch_norm(out, train=is_train, name='bn')
            out=self.hard_swish(out)
            # squeeze and excitation
            if se:
                channel = int(np.shape(out)[-1])
                out = self.squeeze_excitation_layer(out,out_dim=channel, ratio=4, layer_name='se_block')
            return out
    
    def conv2d_NBN_hs(self, input, output_dim, name, bias=False):
        with tf.name_scope(name), tf.compat.v1.variable_scope(name):
            out=self.conv_1x1(input, output_dim, bias=bias, name='pwb')
            out=self.hard_swish(out)
            return out
    
    def squeeze_excitation_layer(self, input, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
    
            squeeze = self.global_avg(input)
    
            excitation = self.Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_excitation1')
            excitation = self.relu6(excitation)
            excitation = self.Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_excitation2')
            excitation = self.hard_sigmoid(excitation)
    
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input * excitation
    
            return scale
    
    def mnv3_block(self, input, k_s, expansion_ratio, output_dim, stride, name, is_train=True, bias=True, shortcut=True, h_swish=False, ratio=16, se=False):
        with tf.compat.v1.variable_scope(name):
            # pw
            bottleneck_dim=expansion_ratio#round(expansion_ratio*input.get_shape().as_list()[-1])
#            print(bottleneck_dim)
            net = self.conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
            net = self.batch_norm(net, train=is_train, name='pw_bn')
            if h_swish:
                    net = self.hard_swish(net)
            else:
                    net = self.relu6(net)
            # dw
            net = self.dwise_conv(net, k_w=k_s, k_h=k_s, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = self.batch_norm(net, train=is_train, name='dw_bn')
            if h_swish:
                    net = self.hard_swish(net)
            else:
                    net = self.relu6(net)
            # squeeze and excitation
            if se:
                    channel = int(np.shape(net)[-1])
                    net = self.squeeze_excitation_layer(net,out_dim=channel, ratio=ratio, layer_name='se_block')
    
            # pw & linear
            net = self.conv_1x1(net, output_dim, name='pw_linear', bias=bias)
            net = self.batch_norm(net, train=is_train, name='pw_linear_bn')
    
            # element wise add, only for stride==1
            if shortcut and stride == 1:
                in_dim=int(input.get_shape().as_list()[-1])
                net_dim = int(net.get_shape().as_list()[-1])
                if in_dim == net_dim:
                    net+=input
                    net = tf.identity(net, name='output')
    
            return net
        
    # def flatten(self, x):
    #     #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    #     return tf.contrib.layers.flatten(x)
    
    def PAM(self, input_tensor, in_dims, name): #############################Temporal
        with tf.compat.v1.variable_scope(name):
            B = input_tensor.shape[0].value
            H = input_tensor.shape[1].value
            W = input_tensor.shape[2].value

            conv1 = self.conv_1x1(input_tensor, in_dims//2, name=name+'_1')                                
            
            conv2 = self.conv_1x1(input_tensor, in_dims//2, name=name+'_2')
            
            conv3 = self.conv_1x1(input_tensor, in_dims, name=name+'_3')
            
            reshape1 = tf.reshape(conv1,(B*H*W,in_dims//2))
            reshape2 = tf.reshape(conv2,(B*H*W,in_dims//2))
            reshape2 = tf.transpose(reshape2,[1,0])
            reshape3 = tf.reshape(conv3,(B*H*W,in_dims))
            
            inner_prodoct1 = tf.matmul(reshape1,reshape2)
            softmax = tf.nn.softmax(inner_prodoct1)
            inner_prodoct2 = tf.matmul(softmax,reshape3)
            reshape4 = tf.reshape(inner_prodoct2,(B,H,W,in_dims))             
            output = tf.add(input_tensor,reshape4)
        return output
    def top_down(self,conv8_1,feature_list_old,s):
        ################################################################### top to down stride=1###################################################################
        feature_list_new = []
        if s==0 :
            feature_list_old = []
            for cnt in range(conv8_1.get_shape().as_list()[1]):  #看feature map的第二維(高256那維)是多少就跑多少次
                feature_list_old.append(tf.expand_dims(conv8_1[:, cnt, :, :], axis=1))  
            print('no')
            ###old第一層加到new第一層###
            feature_list_new.append(tf.expand_dims(conv8_1[:, 0, :, :], axis=1)) 
            with tf.compat.v1.variable_scope("convs_10_1"):
                conv_10_1 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[0], 320, (1, 13), strides=(1, 1), padding='SAME')), 
                                    feature_list_old[1])  #先對第一片做conv，再將做完的結果和原始的第二片相加
                feature_list_new.append(conv_10_1)
        else:
            feature_list_old=feature_list_old
            print('stack')
            ###old第一層加到new第一層###
            feature_list_new.append(tf.expand_dims(conv8_1[:, 0, :, :], axis=1)) 
            with tf.compat.v1.variable_scope("convs_10_1", reuse=True):
                conv_10_1 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[0], 320, (1, 13), strides=(1, 1), padding='SAME')), 
                                    feature_list_old[1])  #先對第一片做conv，再將做完的結果和原始的第二片相加
                feature_list_new.append(conv_10_1) 
        for cnt in range(2, conv8_1.get_shape().as_list()[1]): 
            with tf.compat.v1.variable_scope("convs_10_1", reuse=True):
                conv_10_1 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[cnt - 1], 320, (1, 13), strides=(1, 1), padding='SAME')),
                                    feature_list_old[cnt])
                feature_list_new.append(conv_10_1)  #每次做完都存到feature_list_new(最後會跟原本切片前的大小一樣)
        ###最下層加到第一層###
        # conv_10_1 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[7], 320, (1, 13), strides=(1, 1), padding='SAME')),
        #                             feature_list_old[0])#
        # feature_list_new.append(conv_10_1)
        return feature_list_new
    def down_top(self,conv8_1,feature_list_old,s):
        ################################################################### down to top stride=1 ###################################################################
        feature_list_new = []
        length = int(CFG.TRAIN.IMG_HEIGHT / 32) - 1     
        if s == 0 :
            feature_list_old = []
            for cnt in range(conv8_1.get_shape().as_list()[1]):  #看feature map的第二維(高256那維)是多少就跑多少次
                feature_list_old.append(tf.expand_dims(conv8_1[:, cnt, :, :], axis=1)) 
            print('no') 
            ###old第一層加到new第一層###
            feature_list_new.append(tf.expand_dims(conv8_1[:, length, :, :], axis=1))
            with tf.compat.v1.variable_scope("convs_10_2"):
                conv_10_2 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length], 320, (1, 13), strides=(1, 1), padding='SAME')),
                                    feature_list_old[length - 1])  #對top to down做完的feature map從底部做conv，再將做完的結果和底部的上一片相加
                feature_list_new.append(conv_10_2) 
        else:
            feature_list_old=feature_list_old
            print('stack')
            ###old第一層加到new第一層###
            feature_list_new.append(tf.expand_dims(conv8_1[:, length, :, :], axis=1))
            with tf.compat.v1.variable_scope("convs_10_2", reuse=True):
                conv_10_2 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length], 320, (1, 13), strides=(1, 1), padding='SAME')),
                                    feature_list_old[length - 1])  #對top to down做完的feature map從底部做conv，再將做完的結果和底部的上一片相加
                feature_list_new.append(conv_10_2)
        for cnt in range(2, conv8_1.get_shape().as_list()[1]):  #把剩下的片全部照做
            with tf.compat.v1.variable_scope("convs_10_2", reuse=True):
                conv_10_2 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length-cnt+1], 320, (1, 13), strides=(1, 1), padding='SAME')),
                                feature_list_old[length - cnt])
                feature_list_new.append(conv_10_2)  #每次做完都存到feature_list_new(最後會跟原本切片前的大小一樣)，但是是跟最一開始進SCNN的feature map上下顛倒的
                print('cnt',cnt)
                print('conv_10_1',feature_list_new)
        ###最下層加到第一層###
        # conv_10_2 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[0], 320, (1, 13), strides=(1, 1), padding='SAME')),
        #                         feature_list_old[7])
        # feature_list_new.append(conv_10_2)  #每次做完都存到feature_list_new(最後會跟原本切片前的大小一樣)，但是是跟最一開始進SCNN的feature map上下顛倒的
        feature_list_new.reverse()  #因為上下顛倒，所以做一個倒轉，注意這裡還是一堆片(二維)
        return feature_list_new
    def left_right(self,conv8_1,feature_list_old,s):
        ################################################################### left to right  stride=1###################################################################
        feature_list_new = []
        if s == 0 :
            feature_list_old = []
            for cnt in range(conv8_1.get_shape().as_list()[2]):
                feature_list_old.append(tf.expand_dims(conv8_1[:, :, cnt, :], axis=2))#cnt=15  
            print('no')
            ###old第一層加到new第一層###  
            feature_list_new.append(tf.expand_dims(conv8_1[:, :, 0, :], axis=2))
            with tf.compat.v1.variable_scope("convs_10_3"):
                conv_10_3 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[0], 320, (9, 1), strides=(1, 1), padding='SAME')),
                                    feature_list_old[1])
                feature_list_new.append(conv_10_3)
        else:
            feature_list_old=feature_list_old  
            print('stack')    
            ###old第一層加到new第一層###  
            feature_list_new.append(tf.expand_dims(conv8_1[:, :, 0, :], axis=2))
            with tf.compat.v1.variable_scope("convs_10_3", reuse=True):
                conv_10_3 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[0], 320, (9, 1), strides=(1, 1), padding='SAME')),
                                    feature_list_old[1])
                feature_list_new.append(conv_10_3)
        for cnt in range(2, conv8_1.get_shape().as_list()[2]):
            with tf.compat.v1.variable_scope("convs_10_3", reuse=True):
                conv_10_3 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[cnt-1], 320, (9, 1), strides=(1, 1), padding='SAME')),
                                    feature_list_old[cnt])
                feature_list_new.append(conv_10_3)
        ###最下層加到第一層###
        # conv_10_3 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[15], 320, (9, 1), strides=(1, 1), padding='SAME')),
        #                             feature_list_old[0])
        # feature_list_new.append(conv_10_3) 
        return feature_list_new
    def right_left(self,conv8_1,feature_list_old,s):
        ################################################################### right to left  stride=1###################################################################
        feature_list_new = []
        length = int(CFG.TRAIN.IMG_WIDTH / 32) - 1
        if s == 0 :
            feature_list_old = []
            for cnt in range(conv8_1.get_shape().as_list()[2]):
                feature_list_old.append(tf.expand_dims(conv8_1[:, :, cnt, :], axis=2))#cnt=15  
            print('no')
             ###old第一層加到new第一層###
            feature_list_new.append(tf.expand_dims(conv8_1[:, :, length, :], axis=2))
            with tf.compat.v1.variable_scope("convs_10_4"):
                conv_10_4 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length], 320, (9, 1), strides=(1, 1), padding='SAME')),
                                    feature_list_old[length - 1])
                feature_list_new.append(conv_10_4)
        else:
            feature_list_old=feature_list_old  
            print('stack')
            ###old第一層加到new第一層###
            feature_list_new.append(tf.expand_dims(conv8_1[:, :, length, :], axis=2))
            with tf.compat.v1.variable_scope("convs_10_4", reuse=True):
                conv_10_4 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length], 320, (9, 1), strides=(1, 1), padding='SAME')),
                                    feature_list_old[length - 1])
                feature_list_new.append(conv_10_4)
        for cnt in range(2, conv8_1.get_shape().as_list()[2]):
            with tf.compat.v1.variable_scope("convs_10_4", reuse=True):
                conv_10_4 = tf.add(tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length-cnt-+1], 320, (9, 1), strides=(1, 1), padding='SAME')),
                                    feature_list_old[length - cnt])
                feature_list_new.append(conv_10_4)
        ###最下層加到第一層###
        # conv_10_4 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[0], 320, (9, 1), strides=(1, 1), padding='SAME')),
        #                             feature_list_old[15])
        # feature_list_new.append(conv_10_4)
        feature_list_new.reverse()  
        return feature_list_new
    def slash_top_down(self,conv8_1,feature_list_old,s):
            ################################################################### slash top to dowm =1###################################################################
        feature_list_new = []
        feature_list_old_left = []
        feature_list_old_right = []
        if s == 0 :
            feature_list_old = []
            for cnt in range(conv8_1.get_shape().as_list()[1]): 
                feature_list_old.append(tf.expand_dims(conv8_1[:, cnt, :, :], axis=1))
            ###old前三層加到new前三層###
            feature_list_new.append(feature_list_old[0])
            feature_list_new.append(feature_list_old[1])
            feature_list_new.append(feature_list_old[2]) 
            print('no')
            with tf.compat.v1.variable_scope("convs_10_5"):
                feature_list_old_slice=tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[2], 320, (1, 16), strides=(1, 1), padding='SAME'))
                ###lest###
                for l in range(int(conv8_1.get_shape().as_list()[2]/2)): 
                    feature_list_old_left.append(tf.expand_dims(feature_list_old_slice[:, :, l, :], axis=2))
                ###right###
                for r in range(int(conv8_1.get_shape().as_list()[2]/2),conv8_1.get_shape().as_list()[2]): 
                    feature_list_old_right.append(tf.expand_dims(feature_list_old_slice[:, :, r, :], axis=2))
                ###往左&往右移###
                feature_list_new_left_down = feature_list_old_left[1:8]
                feature_list_new_left_down.insert(7,feature_list_old_left[7])
                feature_list_new_left_down = tf.concat(feature_list_new_left_down,2)

                feature_list_new_right_down = feature_list_old_right[0:7]
                feature_list_new_right_down.insert(0,feature_list_old_right[0])
                feature_list_new_right_down = tf.concat(feature_list_new_right_down,2)
                ###左&右相加###
                feature_list_new_slice = tf.concat([feature_list_new_left_down,feature_list_new_right_down],2)
                conv_10_1=tf.add(feature_list_new_slice,feature_list_old[3])
                feature_list_new.append(conv_10_1)
                
        else:
            feature_list_old=feature_list_old  
            print('stack')
            ###old前三層加到new前三層###
            feature_list_new.append(feature_list_old[0])
            feature_list_new.append(feature_list_old[1])
            feature_list_new.append(feature_list_old[2])
            with tf.compat.v1.variable_scope("convs_10_5",reuse=True):
                feature_list_old_slice=tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[2], 320, (1, 16), strides=(1, 1), padding='SAME'))
                ###lest###
                for l in range(int(conv8_1.get_shape().as_list()[2]/2)): 
                    feature_list_old_left.append(tf.expand_dims(feature_list_old_slice[:, :, l, :], axis=2))
                ###right###
                for r in range(int(conv8_1.get_shape().as_list()[2]/2),conv8_1.get_shape().as_list()[2]): 
                    feature_list_old_right.append(tf.expand_dims(feature_list_old_slice[:, :, r, :], axis=2))
                ###往左&往右移###
                feature_list_new_left_down = feature_list_old_left[1:8]
                feature_list_new_left_down.insert(7,feature_list_old_left[7])
                feature_list_new_left_down = tf.concat(feature_list_new_left_down,2)

                feature_list_new_right_down = feature_list_old_right[0:7]
                feature_list_new_right_down.insert(0,feature_list_old_right[0])
                feature_list_new_right_down = tf.concat(feature_list_new_right_down,2)
                ###左&右相加###
                feature_list_new_slice = tf.concat([feature_list_new_left_down,feature_list_new_right_down],2)
                conv_10_1=tf.add(feature_list_new_slice,feature_list_old[3])
                feature_list_new.append(conv_10_1)
        
        for cnt in range(4, int(conv8_1.get_shape().as_list()[1])):  #把剩下的片全部照做
            with tf.compat.v1.variable_scope("convs_10_5", reuse=True):
                feature_list_old_slice=tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[cnt-1], 320, (1, 16), strides=(1, 1), padding='SAME'))
                ###lest###
                for l in range(int(conv8_1.get_shape().as_list()[2]/2)): 
                    feature_list_old_left.append(tf.expand_dims(feature_list_old_slice[:, :, l, :], axis=2))
                ###right###
                for r in range(int(conv8_1.get_shape().as_list()[2]/2),conv8_1.get_shape().as_list()[2]): 
                    feature_list_old_right.append(tf.expand_dims(feature_list_old_slice[:, :, r, :], axis=2))
                ###往左&往右移###
                feature_list_new_left_down=feature_list_old_left[1:8]
                feature_list_new_left_down.insert(7,feature_list_old_left[7])
                feature_list_new_left_down = tf.concat(feature_list_new_left_down,2)

                feature_list_new_right_down=feature_list_old_right[0:7]
                feature_list_new_right_down.insert(0,feature_list_old_right[0])
                feature_list_new_right_down = tf.concat(feature_list_new_right_down,2)
                ###左&右相加###
                feature_list_new_slice = tf.concat([feature_list_new_left_down,feature_list_new_right_down],2)

                conv_10_1=tf.add(feature_list_new_slice,feature_list_old[cnt])
                feature_list_new.append(conv_10_1) 
        return feature_list_new
    def slash_down_top(self,conv8_1,feature_list_old,s):
            ################################################################### slash down to top =1###################################################################
            feature_list_new = []
            feature_list_old_left = []
            feature_list_old_right = []
            length = int(CFG.TRAIN.IMG_HEIGHT / 32) - 1
            if s==0:
                feature_list_old = []
                print('no')
                for cnt in range(conv8_1.get_shape().as_list()[1]):  #看feature map的第二維(高256那維)是多少就跑多少次
                    feature_list_old.append(tf.expand_dims(conv8_1[:, cnt, :, :], axis=1))  #把每一片feature map存到feature_list_old
                ###old第一層加到new第一層###
                feature_list_new.append(feature_list_old[length])
                with tf.compat.v1.variable_scope("convs_10_6"):
                    feature_list_old_slice=tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length], 320, (1, 16), strides=(1, 1), padding='SAME'))
                    ###lest###
                    for l in range(int(conv8_1.get_shape().as_list()[2]/2)): 
                        feature_list_old_left.append(tf.expand_dims(feature_list_old_slice[:, :, l, :], axis=2))
                    ###right###
                    for r in range(int(conv8_1.get_shape().as_list()[2]/2),conv8_1.get_shape().as_list()[2]): 
                        feature_list_old_right.append(tf.expand_dims(feature_list_old_slice[:, :, r, :], axis=2))
                    ###往左&往右移###
                    feature_list_new_left_down = feature_list_old_left[0:7]
                    feature_list_new_left_down.insert(0,feature_list_old_left[0])
                    feature_list_new_left_down = tf.concat(feature_list_new_left_down,2)

                    feature_list_new_right_down = feature_list_old_right[1:8]
                    feature_list_new_right_down.insert(7,feature_list_old_right[7])
                    feature_list_new_right_down = tf.concat(feature_list_new_right_down,2)
                    ###左&右相加###
                    feature_list_new_slice = tf.concat([feature_list_new_left_down,feature_list_new_right_down],2)

                    conv_10_1=tf.add(feature_list_new_slice,feature_list_old[length-1])
                    feature_list_new.append(conv_10_1)
            else:
                feature_list_old = feature_list_old  
                print('stack')
                print('stack',feature_list_old)
                ###old第一層加到new第一層###
                feature_list_new.append(feature_list_old[length])
                with tf.compat.v1.variable_scope("convs_10_6", reuse=True):
                    feature_list_old_slice=tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length], 320, (1, 16), strides=(1, 1), padding='SAME'))
                    ###lest###
                    for l in range(int(conv8_1.get_shape().as_list()[2]/2)): 
                        feature_list_old_left.append(tf.expand_dims(feature_list_old_slice[:, :, l, :], axis=2))
                    ###right###
                    for r in range(int(conv8_1.get_shape().as_list()[2]/2),conv8_1.get_shape().as_list()[2]): 
                        feature_list_old_right.append(tf.expand_dims(feature_list_old_slice[:, :, r, :], axis=2))
                    ###往左&往右移###
                    feature_list_new_left_down = feature_list_old_left[0:7]
                    feature_list_new_left_down.insert(0,feature_list_old_left[0])
                    feature_list_new_left_down = tf.concat(feature_list_new_left_down,2)

                    feature_list_new_right_down = feature_list_old_right[1:8]
                    feature_list_new_right_down.insert(7,feature_list_old_right[7])
                    feature_list_new_right_down = tf.concat(feature_list_new_right_down,2)
                    ###左&右相加###
                    feature_list_new_slice = tf.concat([feature_list_new_left_down,feature_list_new_right_down],2)

                    conv_10_1=tf.add(feature_list_new_slice,feature_list_old[length-1])
                    feature_list_new.append(conv_10_1)

            for cnt in range(2, int(conv8_1.get_shape().as_list()[1])-2):  #把剩下的片全部照做
                with tf.compat.v1.variable_scope("convs_10_6", reuse=True):
                    feature_list_old_slice=tf.nn.relu(tf.compat.v1.layers.separable_conv2d(feature_list_old[length-cnt+1], 320, (1, 16), strides=(1, 1), padding='SAME'))
                    ###lest###
                    for l in range(int(conv8_1.get_shape().as_list()[2]/2)): 
                        feature_list_old_left.append(tf.expand_dims(feature_list_old_slice[:, :, l, :], axis=2))
                    ###right###
                    for r in range(int(conv8_1.get_shape().as_list()[2]/2),conv8_1.get_shape().as_list()[2]): 
                        feature_list_old_right.append(tf.expand_dims(feature_list_old_slice[:, :, r, :], axis=2))
                    ###往左&往右移###
                    feature_list_new_left_down=feature_list_old_left[1:8]
                    feature_list_new_left_down.insert(7,feature_list_old_left[7])
                    feature_list_new_left_down = tf.concat(feature_list_new_left_down,2)

                    feature_list_new_right_down=feature_list_old_right[0:7]
                    feature_list_new_right_down.insert(0,feature_list_old_right[0])
                    feature_list_new_right_down = tf.concat(feature_list_new_right_down,2)
                    ###左&右相加###
                    feature_list_new_slice = tf.concat([feature_list_new_left_down,feature_list_new_right_down],2)
                    conv_10_1=tf.add(feature_list_new_slice,feature_list_old[length-cnt])
                    feature_list_new.append(conv_10_1)
            ###old加到new第一層###      
            feature_list_new.append(feature_list_old[1])
            feature_list_new.append(feature_list_old[0])
            feature_list_new.reverse() 
            return feature_list_new
        

    def encode(self, input_tensor, name):
        """
        根據MoblieNet_v3框架對輸入的tensor進行編碼
        :param input_tensor:
        :param name:
        :param flags:
        :return: 輸出MoblieNet_v3編碼特徵
        """
        ret = OrderedDict()
        reduction_ratio = 4
        # input_tensor = input_tensor[:,64:256,:]

        with tf.compat.v1.variable_scope(name):
            
            # conv stage 1
            conv1_1 = self.conv2d_block(input_tensor, 16, 3, 2, is_train=True, name='conv1_1', h_swish=True)  # size/2
            
            ret['conv1_1'] = dict()
            ret['conv1_1']['data'] = conv1_1
            ret['conv1_1']['shape'] = conv1_1.get_shape().as_list()  #[2, 128, 256, 16]

            # conv stage 2
            bneck2_1 = self.mnv3_block(conv1_1, 3, 16, 16, 1, is_train=True, name='bneck2_1', h_swish=False, ratio=reduction_ratio, se=False)
            
            ret['bneck2_1'] = dict()
            ret['bneck2_1']['data'] = bneck2_1
            ret['bneck2_1']['shape'] = bneck2_1.get_shape().as_list()  #[2, 128, 256, 16]
            
            output_1 = tf.expand_dims(tf.reduce_sum(tf.abs(bneck2_1)**2, axis=3), axis=3)                            
            
            ret['output_1'] = dict()
            ret['output_1']['data'] = output_1
            ret['output_1']['shape'] = output_1.get_shape().as_list()  #[2, 128, 256, 1]

            # conv stage 3
            bneck3_1 = self.mnv3_block(bneck2_1, 3, 64, 24, 2, is_train=True, name='bneck3_1', h_swish=False, ratio=reduction_ratio, se=False)  # size/4        
            bneck3_2 = self.mnv3_block(bneck3_1, 3, 72, 24, 1, is_train=True, name='bneck3_2', h_swish=False, ratio=reduction_ratio, se=False)
            
            ret['bneck3_2'] = dict()
            ret['bneck3_2']['data'] = bneck3_2
            ret['bneck3_2']['shape'] = bneck3_2.get_shape().as_list()  #fpn_concat_3 [2, 64, 128, 24]
            
            output_2 = tf.image.resize(tf.expand_dims(tf.reduce_sum(tf.abs(bneck3_2)**2, axis=3), axis=3), (128, 256)) 
            
            ret['output_2'] = dict()
            ret['output_2']['data'] = output_2
            ret['output_2']['shape'] = output_2.get_shape().as_list()  #[2, 128, 256, 1]
            
            #self_attention = self.PAM(bneck3_2, 24, name='self_attention')
            #add_function = tf.add(self_attention, bneck3_2)

            # conv stage 4
            bneck4_1 = self.mnv3_block(bneck3_2, 5, 72, 40, 2, is_train=True, name='bneck4_1', h_swish=False, ratio=reduction_ratio, se=True)  # size/8
            bneck4_2 = self.mnv3_block(bneck4_1, 5, 120, 40, 1, is_train=True, name='bneck4_2', h_swish=False, ratio=reduction_ratio, se=True)
            bneck4_3 = self.mnv3_block(bneck4_2, 5, 120, 40, 1, is_train=True, name='bneck4_3', h_swish=False, ratio=reduction_ratio, se=True)
            
            ret['bneck4_3'] = dict()
            ret['bneck4_3']['data'] = bneck4_3
            ret['bneck4_3']['shape'] = bneck4_3.get_shape().as_list()	#fpn_concat_2 [2, 32, 64, 40]
            
            output_3 = tf.image.resize(tf.expand_dims(tf.reduce_sum(tf.abs(bneck4_3)**2, axis=3), axis=3), (128, 256))
            
            ret['output_3'] = dict()
            ret['output_3']['data'] = output_3
            ret['output_3']['shape'] = output_3.get_shape().as_list()  #[2, 128, 256, 1]

            # conv stage 5
            bneck5_1 = self.mnv3_block(bneck4_3, 3, 240, 80, 2, is_train=True, name='bneck5_1', h_swish=True, ratio=reduction_ratio, se=False) # size/16       
            bneck5_2 = self.mnv3_block(bneck5_1, 3, 200, 80, 1, is_train=True, name='bneck5_2', h_swish=True, ratio=reduction_ratio, se=False)         
            bneck5_3 = self.mnv3_block(bneck5_2, 3, 184, 80, 1, is_train=True, name='bneck5_3', h_swish=True, ratio=reduction_ratio, se=False)
            bneck5_4 = self.mnv3_block(bneck5_3, 3, 184, 80, 1, is_train=True, name='bneck5_4', h_swish=True, ratio=reduction_ratio, se=False)
            
            ret['bneck5_4'] = dict()
            ret['bneck5_4']['data'] = bneck5_4
            ret['bneck5_4']['shape'] = bneck5_4.get_shape().as_list()  #[2, 16, 32, 80]

            # conv_stage 6
            bneck6_1 = self.mnv3_block(bneck5_4, 3, 480, 112, 1, is_train=True, name='bneck6_1', h_swish=True, ratio=reduction_ratio, se=True)
            bneck6_2 = self.mnv3_block(bneck6_1, 3, 672, 112, 1, is_train=True, name='bneck6_2', h_swish=True, ratio=reduction_ratio, se=True)
            
            ret['bneck6_2'] = dict()
            ret['bneck6_2']['data'] = bneck6_2
            ret['bneck6_2']['shape'] = bneck6_2.get_shape().as_list()  #fpn_concat_1 [2, 16, 32, 112]
            
            output_4 = tf.expand_dims(tf.reduce_sum(tf.abs(bneck6_2)**2, axis=3), axis=3)      
            
            ret['output_4'] = dict()
            ret['output_4']['data'] = output_4
            ret['output_4']['shape'] = output_4.get_shape().as_list()  #[2, 16, 32, 1]
            
            # conv stage 7
            bneck7_1 = self.mnv3_block(bneck6_2, 5, 672, 160, 2, is_train=True, name='bneck7_1', h_swish=True, ratio=reduction_ratio, se=True) # size/32
            bneck7_2 = self.mnv3_block(bneck7_1, 5, 960, 160, 1, is_train=True, name='bneck7_2', h_swish=True, ratio=reduction_ratio, se=True)
            bneck7_3 = self.mnv3_block(bneck7_2, 5, 960, 160, 1, is_train=True, name='bneck7_3', h_swish=True, ratio=reduction_ratio, se=True) 
            
            ret['bneck7_3'] = dict()
            ret['bneck7_3']['data'] = bneck7_3
            ret['bneck7_3']['shape'] = bneck7_3.get_shape().as_list()  #[2, 8, 16, 160]
            
            output_5 = tf.image.resize(tf.expand_dims(tf.reduce_sum(tf.abs(bneck7_3)**2, axis=3), axis=3), (16, 32))
            
            ret['output_5'] = dict()
            ret['output_5']['data'] = output_5
            ret['output_5']['shape'] = output_5.get_shape().as_list()  #[2, 16, 32, 1]
            
            # conv stage 8
            conv8_1 = self.conv2d_hs(bneck7_3, 320, is_train=True, name='conv8_1')
            
            ret['conv8_1'] = dict()
            ret['conv8_1']['data'] = conv8_1
            ret['conv8_1']['shape'] = conv8_1.get_shape().as_list()  #[2, 8, 16, 320]
            print('111111111111111',conv8_1.shape[3])
            

            reshape_hidden = tf.reshape(conv8_1, (conv8_1.shape[3], (conv8_1.shape[0])*(conv8_1.shape[1])*(conv8_1.shape[2])))
                ################################################################### top to down stride=1###################################################################
            feature_list_new = self.top_down(conv8_1,conv8_1,s=0)
            processed_feature_top = tf.stack(feature_list_new, axis=1)
            processed_feature_top = tf.squeeze(processed_feature_top, axis=2)
            reshape_top = tf.reshape(processed_feature_top, ((processed_feature_top.shape[0])*(processed_feature_top.shape[1])*(processed_feature_top.shape[2]), processed_feature_top.shape[3]))

            #   ################################################################### down to top stride=1 ###################################################################
            feature_list_new = self.down_top(conv8_1,feature_list_new,s=0)
            processed_feature_dowm = tf.stack(feature_list_new, axis=1)  #將一堆片以高度的維度stack，stack完後會長(B,H,1,W,C)
            processed_feature_dowm = tf.squeeze(processed_feature_dowm, axis=2)  #須把第2維的1給squeeze，才會回到(B,H,W,C)
            reshape_down = tf.reshape(processed_feature_dowm, ((processed_feature_dowm.shape[0])*(processed_feature_dowm.shape[1])*(processed_feature_dowm.shape[2]), processed_feature_dowm.shape[3]))
            # print('')
            # print('reshape_down',reshape_down)
            # print('')
                ################################################################### left to right  stride=1###################################################################
            feature_list_new = self.left_right(conv8_1,feature_list_new,s=0)
            processed_feature_left = tf.stack(feature_list_new, axis=2)
            processed_feature_left = tf.squeeze(processed_feature_left, axis=3)
            reshape_left = tf.reshape(processed_feature_left, ((processed_feature_left.shape[0])*(processed_feature_left.shape[1])*(processed_feature_left.shape[2]), processed_feature_left.shape[3])) 
            # print('')
            # print('reshape_left',reshape_left)
            # print('')
                ################################################################### right to left stride=1###################################################################
            feature_list_new = self.right_left(conv8_1,feature_list_new,s=0)
            processed_feature_right = tf.stack(feature_list_new, axis=2)
            processed_feature_right = tf.squeeze(processed_feature_right, axis=3)
            reshape_right = tf.reshape(processed_feature_right, ((processed_feature_right.shape[0])*(processed_feature_right.shape[1])*(processed_feature_right.shape[2]), processed_feature_right.shape[3]))
            # print('')
            # print('reshape_right',reshape_right)
            # print('')
                ################################################################### slash top to dowm =1###################################################################
            feature_list_new = self.slash_top_down(conv8_1,feature_list_new,s=0)          
            # result = tf.concat(feature_list_new,1)
            processed_feature_slashdown = tf.stack(feature_list_new, axis=1)
            processed_feature_slashdown = tf.squeeze(processed_feature_slashdown, axis=2)
            reshape_slashdown = tf.reshape(processed_feature_slashdown, ((processed_feature_slashdown.shape[0])*(processed_feature_slashdown.shape[1])*(processed_feature_slashdown.shape[2]), processed_feature_slashdown.shape[3]))
                ################################################################### slash  dowm to top =1###################################################################
            feature_list_new = self.slash_down_top(conv8_1,feature_list_new,s=0)
            # result = tf.concat(feature_list_new,1)
            processed_feature_slashtop = tf.stack(feature_list_new, axis=1)
            processed_feature_slashtop = tf.squeeze(processed_feature_slashtop, axis=2)
            reshape_slashtop = tf.reshape(processed_feature_slashtop, ((processed_feature_slashtop.shape[0])*(processed_feature_slashtop.shape[1])*(processed_feature_slashtop.shape[2]), processed_feature_slashtop.shape[3]))

            
            inner_product_top = tf.matmul(reshape_hidden, reshape_top)
            softmax_top = tf.nn.softmax(inner_product_top)
            inner_product_down = tf.matmul(reshape_down, softmax_top)
            inner_product_down = tf.transpose(inner_product_down, [1, 0])
            
            inner_product_left = tf.matmul(inner_product_down, reshape_left)
            softmax_left = tf.nn.softmax(inner_product_left)
            inner_product_right = tf.matmul(reshape_right, softmax_left)
            inner_product_right = tf.transpose(inner_product_right, [1,0])
            
            inner_product_slashdown = tf.matmul(inner_product_right, reshape_slashdown)
            softmax_slashdown = tf.nn.softmax(inner_product_slashdown)
            inner_product_slashtop = tf.matmul(reshape_slashtop, softmax_slashdown)
            inner_product_slashtop = tf.transpose(inner_product_slashtop, [1,0])
            feature_sum = tf.math.add(inner_product_slashtop, reshape_hidden)
            
            
            result = tf.reshape(feature_sum, (conv8_1.shape[0], conv8_1.shape[1], conv8_1.shape[2], conv8_1.shape[3]))
                    
            ret['result'] = dict()
            ret['result']['data'] = result
            ret['result']['shape'] = result.get_shape().as_list()
            
            # # add SCNN message passing part #
            #     # top to down #
            # feature_list_old = []
            # feature_list_new = []
            # for cnt in range(conv8_1.get_shape().as_list()[1]):
            #     feature_list_old.append(tf.expand_dims(conv8_1[:, cnt, :, :], axis=1))
            # feature_list_new.append(tf.expand_dims(conv8_1[:, 0, :, :], axis=1))
            
            # with tf.compat.v1.variable_scope("convs_10_1"):
            #     conv_10_1 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[0], 320, (1, 9), strides=(1, 1), padding='SAME')),
            #                       feature_list_old[1])
            #     feature_list_new.append(conv_10_1)

            # for cnt in range(2, conv8_1.get_shape().as_list()[1]):
            #     with tf.compat.v1.variable_scope("convs_10_1", reuse=True):
            #         conv_10_1 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_new[cnt - 1], 320, (1, 9), strides=(1, 1), padding='SAME')),
            #                           feature_list_old[cnt])
            #         feature_list_new.append(conv_10_1)
            
            #     # down to top #
            # feature_list_old = feature_list_new
            # feature_list_new = []
            # length = int(CFG.TRAIN.IMG_HEIGHT / 32) - 1
            # feature_list_new.append(feature_list_old[length])
            
            # with tf.compat.v1.variable_scope("convs_10_2"):
            #     conv_10_2 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[length], 320, (1, 9), strides=(1, 1), padding='SAME')),
            #                       feature_list_old[length - 1])
            #     feature_list_new.append(conv_10_2)

            # for cnt in range(2, conv8_1.get_shape().as_list()[1]):
            #     with tf.compat.v1.variable_scope("convs_10_2", reuse=True):
            #         conv_10_2 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_new[cnt - 1], 320, (1, 9), strides=(1, 1), padding='SAME')),
            #                       feature_list_old[length - cnt])
            #         feature_list_new.append(conv_10_2)

            # feature_list_new.reverse()

            # processed_feature = tf.stack(feature_list_new, axis=1)
            # processed_feature = tf.squeeze(processed_feature, axis=2)
            
            #     # left to right #
            # feature_list_old = []
            # feature_list_new = []
            # for cnt in range(processed_feature.get_shape().as_list()[2]):
            #     feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
            # feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

            # with tf.compat.v1.variable_scope("convs_10_3"):
            #     conv_10_3 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[0], 320, (9, 1), strides=(1, 1), padding='SAME')),
            #                       feature_list_old[1])
            #     feature_list_new.append(conv_10_3)

            # for cnt in range(2, processed_feature.get_shape().as_list()[2]):
            #     with tf.compat.v1.variable_scope("convs_10_3", reuse=True):
            #         conv_10_3 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_new[cnt - 1], 320, (9, 1), strides=(1, 1), padding='SAME')),
            #                           feature_list_old[cnt])
            #         feature_list_new.append(conv_10_3)
                    
            #     # right to left #
            # feature_list_old = feature_list_new
            # feature_list_new = []
            # length = int(CFG.TRAIN.IMG_WIDTH / 32) - 1
            # feature_list_new.append(feature_list_old[length])
            
            # with tf.compat.v1.variable_scope("convs_10_4"):
            #     conv_10_4 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_old[length], 320, (9, 1), strides=(1, 1), padding='SAME')),
            #                       feature_list_old[length - 1])
            #     feature_list_new.append(conv_10_4)

            # for cnt in range(2, processed_feature.get_shape().as_list()[2]):
            #     with tf.compat.v1.variable_scope("convs_10_4", reuse=True):
            #         conv_10_4 = tf.add(tf.nn.relu(tf.layers.separable_conv2d(feature_list_new[cnt - 1], 320, (9, 1), strides=(1, 1), padding='SAME')),
            #                           feature_list_old[length - cnt])
            #         feature_list_new.append(conv_10_4)

            # feature_list_new.reverse()
            # processed_feature = tf.stack(feature_list_new, axis=2)
            # processed_feature = tf.squeeze(processed_feature, axis=3)
            
            # ret['processed_feature'] = dict()
            # ret['processed_feature']['data'] = processed_feature
            # ret['processed_feature']['shape'] = processed_feature.get_shape().as_list()

            
        return ret
    # TODO(luoyao) luoyao@baidu.com 檢查batch normalization分佈和遷移是否合理


if __name__ == '__main__':
    a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 256, 512, 3], name='input')
    encoder = mobilenet_v3_large_Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
