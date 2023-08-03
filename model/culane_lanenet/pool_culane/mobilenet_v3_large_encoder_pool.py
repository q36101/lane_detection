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
# import tensorflow.nn as nn
tf.compat.v1.disable_eager_execution()
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (Layer,Input,LayerNormalization,
                                    Dense,Dropout,Conv2D,AveragePooling2D)

# from config import cnn_basenet_swin
from encoder_decoder_model import cnn_basenet_swin
# import cnn_basenet_swin
from einops import rearrange
from config import global_config
# from group import GroupNorm


CFG = global_config.cfg
weight_decay=1e-5
def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf
class Pooling(Layer):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = AveragePooling2D(pool_size=pool_size,strides=(1,1),padding='same')
    def forward(self, x):
        return self.pool(x) - x
    
class Mlp(Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2D(filters=in_features , kernel_size=1, padding="valid")
        self.fc2 = Conv2D(filters=out_features , kernel_size=1, padding="valid")
        self.drop = Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def GroupNorm(x, G, eps=1e-5):
    N, H, W, C = x.shape
    x = tf.reshape(x, shape=[N, H, W, G, C//G])
    mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var+eps)
    x = tf.reshape(x, shape=[N, H, W, C])
    return x #* gamma + beta
class PoolFormerBlock(Layer):
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=gelu, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = tf.Variable(
                layer_scale_init_value * tf.ones([2,1,1,dim]))
            self.layer_scale_2 = tf.Variable(
                layer_scale_init_value * tf.ones([2,1,1,dim]))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + (
                self.layer_scale_1* self.token_mixer(GroupNorm(x)))
            x = x + (
                self.layer_scale_2* self.mlp(GroupNorm(x)))
        else:
            x = x + self.token_mixer(GroupNorm(x))
            x = x + self.mlp(GroupNorm(x))
        return x
class mobilenet_v3_large_Encoder(cnn_basenet_swin.CNNBaseModel):
    """
    實現了一個基於MoblieNet_v3的特徵編碼類
    """
    def __init__(self ,phase):
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
            
            
            conv1_1 = self.conv2d_block(input_tensor, 16, 3, 2, is_train=True, name='conv1_1', h_swish=True)  # size/2
            
            ret['conv1_1'] = dict()
            ret['conv1_1']['data'] = conv1_1
            ret['conv1_1']['shape'] = conv1_1.get_shape().as_list()  #[2, 128, 256, 16]

            ###POOL_FORMER-1####
            x = conv1_1
            print('conv1_1',x,type(x))
            N, H, W, C = x.shape
            kernel_size=16
            filters=int(kernel_size*kernel_size*C)
            x = Conv2D(filters=filters , kernel_size=kernel_size , strides=kernel_size  , padding="same")(x)
            x = PoolFormerBlock(dim=filters, pool_size=3, mlp_ratio=4., act_layer=gelu, norm_layer=GroupNorm, drop=0.,use_layer_scale=True, layer_scale_init_value=1e-5)(x)
            x = rearrange(x ,'b h w  (p1 p2 c) -> b (h p1) (w p2) c',p1=int(kernel_size),p2=int(kernel_size))
            # x = tf.concat([y,x],1)
            conv1_1 = x

            # conv stage 2
            bneck2_1 = self.mnv3_block(conv1_1, 3, 16, 16, 1, is_train=True, name='bneck2_1', h_swish=False, ratio=reduction_ratio, se=False)
            
            ret['bneck2_1'] = dict()
            ret['bneck2_1']['data'] = bneck2_1
            ret['bneck2_1']['shape'] = bneck2_1.get_shape().as_list()  #[2, 128, 256, 16]
            
            output_1 = tf.expand_dims(tf.reduce_mean(tf.abs(bneck2_1)**2, axis=3), axis=3)                            
            output_1 = tf.compat.v1.layers.batch_normalization(output_1)                           
            
            ret['output_1'] = dict()
            ret['output_1']['data'] = output_1
            ret['output_1']['shape'] = output_1.get_shape().as_list()  #[2, 128, 256, 1]

            # conv stage 3
            bneck3_1 = self.mnv3_block(bneck2_1, 3, 64, 24, 2, is_train=True, name='bneck3_1', h_swish=False, ratio=reduction_ratio, se=False)  # size/4        
            bneck3_2 = self.mnv3_block(bneck3_1, 3, 72, 24, 1, is_train=True, name='bneck3_2', h_swish=False, ratio=reduction_ratio, se=False)
            
            ###POOL_FORMER-2####
            x = bneck3_2
            print('bneck3_2',x,type(x))
            N, H, W, C = x.shape
            # y = x[:,0:int(H/4),:,:]
            # x = x[:,int(H/4):,:,:]
            kernel_size=16
            filters=int(kernel_size*kernel_size*C)
            x = Conv2D(filters=filters , kernel_size=kernel_size , strides=kernel_size  , padding="same")(x)
            print('x',x)
            x = PoolFormerBlock(dim=filters, pool_size=3, mlp_ratio=4., act_layer=gelu, norm_layer=GroupNorm, drop=0.,use_layer_scale=True, layer_scale_init_value=1e-5)(x)
            print('x',x)
            x = rearrange(x ,'b h w  (p1 p2 c) -> b (h p1) (w p2) c',p1=int(kernel_size),p2=int(kernel_size))
            print('x',x)
            # x = tf.concat([y,x],1)
            bneck3_2 = x
            
            ret['bneck3_2'] = dict()
            ret['bneck3_2']['data'] = bneck3_2
            ret['bneck3_2']['shape'] = bneck3_2.get_shape().as_list()  #fpn_concat_3 [2, 64, 128, 24]

            print('bneck3_2',bneck3_2,type(bneck3_2))
            
            output_2 = tf.image.resize(tf.expand_dims(tf.reduce_mean(tf.abs(bneck3_2)**2, axis=3), axis=3), (128, 256)) 
            output_2 = tf.compat.v1.layers.batch_normalization(output_2)
            
            ret['output_2'] = dict()
            ret['output_2']['data'] = output_2
            ret['output_2']['shape'] = output_2.get_shape().as_list()  #[2, 128, 256, 1]

            # conv stage 4
            bneck4_1 = self.mnv3_block(bneck3_2, 5, 72, 40, 2, is_train=True, name='bneck4_1', h_swish=False, ratio=reduction_ratio, se=True)  # size/8
            bneck4_2 = self.mnv3_block(bneck4_1, 5, 120, 40, 1, is_train=True, name='bneck4_2', h_swish=False, ratio=reduction_ratio, se=True)
            bneck4_3 = self.mnv3_block(bneck4_2, 5, 120, 40, 1, is_train=True, name='bneck4_3', h_swish=False, ratio=reduction_ratio, se=True)
            
            ###POOL_FORMER-2####
            x = bneck4_3
            print('bneck4_3',x,type(x))
            N, H, W, C = x.shape
            # y = x[:,0:int(H/4),:,:]
            # x = x[:,int(H/4):,:,:]
            print('x',x)
            kernel_size=16
            filters=int(kernel_size*kernel_size*C)
            x = Conv2D(filters=filters , kernel_size=kernel_size , strides=kernel_size  , padding="same")(x)
            print('x',x)
            x = PoolFormerBlock(dim=filters, pool_size=3, mlp_ratio=4., act_layer=gelu, norm_layer=GroupNorm, drop=0.,use_layer_scale=True, layer_scale_init_value=1e-5)(x)
            print('x',x)
            x = rearrange(x ,'b h w  (p1 p2 c) -> b (h p1) (w p2) c',p1=int(kernel_size),p2=int(kernel_size))
            print('x',x)
            # x = tf.concat([y,x],1)
            bneck4_3 = x
            
            ret['bneck4_3'] = dict()
            ret['bneck4_3']['data'] = bneck4_3
            ret['bneck4_3']['shape'] = bneck4_3.get_shape().as_list()	#fpn_concat_2 [2, 32, 64, 40]
            
            output_3 = tf.image.resize(tf.expand_dims(tf.reduce_mean(tf.abs(bneck4_3)**2, axis=3), axis=3), (128, 256))
            output_3 = tf.compat.v1.layers.batch_normalization(output_3) 
            
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
            
            output_4 = tf.image.resize(tf.expand_dims(tf.reduce_mean(tf.abs(bneck6_2)**2, axis=3), axis=3) , (128, 256))     
            output_4 = tf.compat.v1.layers.batch_normalization(output_4)     
            
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
            
            output_5 = tf.image.resize(tf.expand_dims(tf.reduce_mean(tf.abs(bneck7_3)**2, axis=3), axis=3), (128, 256))
            output_5 = tf.compat.v1.layers.batch_normalization(output_5)
            
            ret['output_5'] = dict()
            ret['output_5']['data'] = output_5
            ret['output_5']['shape'] = output_5.get_shape().as_list()  #[2, 16, 32, 1]
            
            # conv stage 8
            conv8_1 = self.conv2d_hs(bneck7_3, 320, is_train=True, name='conv8_1')
            
            ret['conv8_1'] = dict()
            ret['conv8_1']['data'] = conv8_1
            ret['conv8_1']['shape'] = conv8_1.get_shape().as_list()  #[2, 8, 16, 320]

            result = conv8_1

            # reshape_hidden = tf.compat.v1.reshape(conv8_1, (conv8_1.shape[3], (conv8_1.shape[0])*(conv8_1.shape[1])*(conv8_1.shape[2]))   )     
            #     ################################################################### top to down ###################################################################
            # feature_list_new = []
            # feature_list_old = []
            # for cnt in range(conv8_1.get_shape().as_list()[1]):  #看feature map的第二維(高256那維)是多少就跑多少次
            #     feature_list_old.append(tf.expand_dims(conv8_1[:, cnt, :, :], axis=1))  
            # print('no')
            # ###old第一層加到new第一層###
            # feature_list_new.append(tf.expand_dims(conv8_1[:, 0, :, :], axis=1)) 
            # with tf.compat.v1.variable_scope("convs_10_1"):

            #     conv_10_1 = tf.add(feature_list_old[0],feature_list_old[1])
            #     conv_10_1_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_1, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_1_add)

            #     conv_10_1 = tf.add(conv_10_1,feature_list_old[2])
            #     conv_10_1_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_1, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_1_add)

            # for cnt in range(3, conv8_1.get_shape().as_list()[1]): 
            #     with tf.compat.v1.variable_scope("convs_10_1", reuse=True):
            #         print('cnt=',cnt,"+",cnt-1,"+",cnt-2)
            #         print('')
            #         conv_10_1=tf.subtract(tf.add(conv_10_1,feature_list_old[cnt]),feature_list_old[cnt-3])
            #         conv_10_1_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_1, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #         feature_list_new.append(conv_10_1_add)
            # print('top to down=',feature_list_new)
                    
            # processed_feature_top = tf.stack(feature_list_new, axis=1)
            # processed_feature_top = tf.squeeze(processed_feature_top, axis=2)
            
            # reshape_top = tf.reshape(processed_feature_top, ((processed_feature_top.shape[0])*(processed_feature_top.shape[1])*(processed_feature_top.shape[2]), processed_feature_top.shape[3]))
            
            #     ################################################################### down to top ###################################################################
            # # feature_list_old = feature_list_old
            # feature_list_new = []
            # length = int(CFG.TRAIN.IMG_HEIGHT / 32) - 1
            # print('length',length)
            # feature_list_new.append(feature_list_old[length])
            
            # with tf.compat.v1.variable_scope("convs_10_2"):

            #     conv_10_2 = tf.add(feature_list_old[length],feature_list_old[length-1])
            #     conv_10_2_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_2, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_2_add)

            #     conv_10_2 = tf.add(conv_10_2,feature_list_old[length-2])
            #     conv_10_2_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_2, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_2_add)

            # for cnt in range(3, conv8_1.get_shape().as_list()[1]):  #把剩下的片全部照做
            #     with tf.compat.v1.variable_scope("convs_10_2", reuse=True):
            #         print('cnt=',length-cnt,"+",length-(cnt-1),"+",length-(cnt-2))
            #         print('')
            #         conv_10_2=tf.subtract(tf.add(conv_10_2,feature_list_old[cnt]),feature_list_old[length-(cnt-3)])
            #         conv_10_2_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_2, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #         feature_list_new.append(conv_10_2_add)
            # print('down to top=',feature_list_new)

            # feature_list_new.reverse()  #因為上下顛倒，所以做一個倒轉，注意這裡還是一堆片(二維)

            # processed_feature_dowm = tf.stack(feature_list_new, axis=1)  #將一堆片以高度的維度stack，stack完後會長(B,H,1,W,C)
            # processed_feature_dowm = tf.squeeze(processed_feature_dowm, axis=2)  #須把第2維的1給squeeze，才會回到(B,H,W,C)
            
            # #conv_top_down = self.conv_1x1(processed_feature_dowm, 160, name='conv_top_down')
            # reshape_down = tf.reshape(processed_feature_dowm, ((processed_feature_dowm.shape[0])*(processed_feature_dowm.shape[1])*(processed_feature_dowm.shape[2]), processed_feature_dowm.shape[3]))
            
            #     ################################################################### left to right ###################################################################
            # feature_list_old = []
            # feature_list_new = []
            # for cnt in range(conv8_1.get_shape().as_list()[2]):
            #     feature_list_old.append(tf.expand_dims(conv8_1[:, :, cnt, :], axis=2))
            # feature_list_new.append(tf.expand_dims(conv8_1[:, :, 0, :], axis=2))

            # with tf.compat.v1.variable_scope("convs_10_3"):
            #     conv_10_3 = tf.add(feature_list_old[0],feature_list_old[1])
            #     conv_10_3_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_3, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_3_add)

            #     conv_10_3 = tf.add(conv_10_3,feature_list_old[2])
            #     conv_10_3_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_3, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_3_add)

            # for cnt in range(3, conv8_1.get_shape().as_list()[2]):
            #     with tf.compat.v1.variable_scope("convs_10_3", reuse=True):
            #         print('cnt=',cnt,"+",cnt-1,"+",cnt-2)
            #         print('')
            #         conv_10_3=tf.subtract(tf.add(conv_10_3,feature_list_old[cnt]),feature_list_old[cnt-3])
            #         conv_10_3_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_3, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #         feature_list_new.append(conv_10_3_add)
            # print('left to right',feature_list_new)
                    
            # processed_feature_left = tf.stack(feature_list_new, axis=2)
            # processed_feature_left = tf.squeeze(processed_feature_left, axis=3)
            
            # reshape_left = tf.reshape(processed_feature_left, ((processed_feature_left.shape[0])*(processed_feature_left.shape[1])*(processed_feature_left.shape[2]), processed_feature_left.shape[3]))
                    
            #     ################################################################### right to left ###################################################################
            # # feature_list_old = feature_list_old
            # feature_list_new = []
            # length = int(CFG.TRAIN.IMG_WIDTH / 32) - 1
            # feature_list_new.append(feature_list_old[length])
            
            # with tf.compat.v1.variable_scope("convs_10_4"):
            #     conv_10_4 = tf.add(feature_list_old[length],feature_list_old[length-1])
            #     conv_10_4_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_4, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_4_add)

            #     conv_10_4 = tf.add(conv_10_4,feature_list_old[length-2])
            #     conv_10_4_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_4, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #     feature_list_new.append(conv_10_4_add)
            # print('length=',length)
            # for cnt in range(3, conv8_1.get_shape().as_list()[2]):
            #     with tf.compat.v1.variable_scope("convs_10_4", reuse=True):
            #         print('cnt=',length-cnt,"+",length-(cnt-1),"+",length-(cnt-2))
            #         print('')
            #         conv_10_4=tf.subtract(tf.add(conv_10_4,feature_list_old[cnt]),feature_list_old[length-(cnt-3)])
            #         conv_10_4_add = tf.nn.relu(tf.compat.v1.layers.separable_conv2d(conv_10_4, 320, (1, 9), strides=(1, 1), padding='SAME'))
            #         feature_list_new.append(conv_10_4_add)

            # feature_list_new.reverse()
            # processed_feature_right = tf.stack(feature_list_new, axis=2)
            # processed_feature_right = tf.squeeze(processed_feature_right, axis=3)
            
            # #conv_left_right = self.conv_1x1(processed_feature, 160, name='conv_left_right')
            # reshape_right = tf.reshape(processed_feature_right, ((processed_feature_right.shape[0])*(processed_feature_right.shape[1])*(processed_feature_right.shape[2]), processed_feature_right.shape[3]))
            # #reshape_right = tf.transpose(reshape_right, [1,0])
            
            # inner_product_top = tf.matmul(reshape_hidden, reshape_top)
            # softmax_top = tf.nn.softmax(inner_product_top)
            # inner_product_down = tf.matmul(reshape_down, softmax_top)
            # inner_product_down = tf.transpose(inner_product_down, [1, 0])
            # inner_product_left = tf.matmul(inner_product_down, reshape_left)
            # softmax_left = tf.nn.softmax(inner_product_left)
            # inner_product_right = tf.matmul(reshape_right, softmax_left)
            # inner_product_right = tf.transpose(inner_product_right, [1,0])
            # feature_sum = tf.math.add(inner_product_right, reshape_hidden)
            # result = tf.reshape(feature_sum, (conv8_1.shape[0], conv8_1.shape[1], conv8_1.shape[2], conv8_1.shape[3]))
                    
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
            # keras.utils.plot_model()
            
        return ret
    # TODO(luoyao) luoyao@baidu.com 檢查batch normalization分佈和遷移是否合理


if __name__ == '__main__':
    a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 256, 512, 3], name='input')
    encoder = mobilenet_v3_large_Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
            #dim=2048
            # x = GroupNorm(x,dim)
            # print('GroupNorm',x,type(x))
            # AveragePoolingx = AveragePooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(x)
            # print('AveragePooling2D',x,type(x))
            # x = AveragePoolingx-x
            # print('AveragePooling2D',x,type(x))
            # x = Conv2D(filters=64 , kernel_size=1, padding="valid")(x)
            # x = gelu(x)
            # x = Dropout(rate=0.)(x)
            # x = Conv2D(filters=768 , kernel_size=1, padding="valid")(x)
            # print('AveragePooling2D',x,type(x))

            # layer_scale_init_value=1e-5
            # use_layer_scale='True'
            # if use_layer_scale:
            #     layer_scale_1 = tf.Variable(layer_scale_init_value * tf.ones([2,1,1,dim]))
            #     layer_scale_2 = tf.Variable(layer_scale_init_value * tf.ones([2,1,1,dim]))
            # print('layer_scale_1',tf.ones([dim,1,1]))
            # print('layer_scale_1',layer_scale_1)
            # print('layer_scale_2',layer_scale_2)

            # if use_layer_scale:
            #     x = GroupNorm(x,dim)
            #     x = AveragePooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(x) -x
            #     x1 = x + layer_scale_1* x

            #     x = GroupNorm(x1,dim)
            #     x = Conv2D(filters=4096, kernel_size=1, padding="valid")(x)
            #     x = gelu(x)
            #     x = Dropout(rate=0.)(x)
            #     x = Conv2D(filters=2048 , kernel_size=1, padding="valid")(x)

            #     x = x1 + layer_scale_2* x
            # else:
            #     x = x + self.drop_path(self.token_mixer(GroupNorm(x,dim)))
            #     x = x + self.drop_path(self.mlp(GroupNorm(x,dim)))
