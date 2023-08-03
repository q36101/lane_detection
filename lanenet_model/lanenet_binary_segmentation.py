#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_binary_segmentation.py
# @IDE: PyCharm Community Edition
"""
實現LaneNet中的二分類圖像分割模型
"""
import tensorflow as tf

from encoder_decoder_model import Inception_ResNet_v2_encoder
from encoder_decoder_model import Inception_ResNet_v2_decoder
from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import cnn_basenet


class LaneNetBinarySeg(cnn_basenet.CNNBaseModel):
    """
    實現語義分割模型
    """
    def __init__(self, phase, net_flag):
        """

        """
        super(LaneNetBinarySeg, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
            self._decoder = fcn_decoder.FCNDecoder()

        elif self._net_flag == 'irv2':
            self._encoder = Inception_ResNet_v2_encoder.Inception_ResNet_v2_Encoder(phase=phase)
            self._decoder = Inception_ResNet_v2_decoder.Inception_ResNet_v2_Decoder()
        
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret

            elif self._net_flag.lower() == 'irv2':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dropout_output'])    
                return decode_ret

    def compute_loss(self, input_tensor, label, name):
        """
        計算損失函數
        :param input_tensor:
        :param label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # 计算损失
            decode_logits = inference_ret['logits']
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decode_logits, labels=tf.squeeze(label, squeeze_dims=[3]),
                name='entropy_loss')

            ret = dict()
            ret['entropy_loss'] = loss
            ret['inference_logits'] = inference_ret['logits']

            return ret
