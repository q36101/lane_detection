#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午2:38
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : dilation_decoder.py
# @IDE: PyCharm Community Edition
"""
實現一個全卷積網絡解碼類
"""
import tensorflow as tf

from encoder_decoder_model import cnn_basenet_swin
from encoder_decoder_model import vgg_encoder_swin


class FCNDecoder(cnn_basenet_swin.CNNBaseModel):
    """
    實現一個全卷積解碼類
    """
    def __init__(self):
        """

        """
        super(FCNDecoder, self).__init__()

    def decode(self, input_tensor_dict, decode_layer_list, name):#, fpn_concat_list, attention_map_list):
        """
        解碼特徵信息反捲積還原
        :param input_tensor_dict:
        :param decode_layer_list: 需要解碼的層名稱需要由深到淺順序寫
                                  eg. ['pool5', 'pool4', 'pool3']
        :param attention_map_list: ['output_5', 'output_4', 'output_3', 'output_2', 'output_1']                          
        :param name:
        :return:
        """
        ret = dict()

        with tf.compat.v1.variable_scope(name):
            
#            sad_list = []
#            for j in range(len(attention_map_list)):
#                sad_list.append(input_tensor_dict[attention_map_list[j]]['data'])
#                
#            # score stage 1
#            input_tensor = input_tensor_dict[decode_layer_list[0]]['data']
#            
#            score = self.conv2d(inputdata=input_tensor, out_channel=64,
#                                kernel_size=1, use_bias=False, name='score_origin')
#            decode_layer_list = decode_layer_list[1:]
#            for i in range(len(decode_layer_list)):
#                deconv = self.deconv2d(inputdata=score, out_channel=64, kernel_size=4,
#                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))
#
#                input_tensor = input_tensor_dict[decode_layer_list[i]]['data']
#                input_fpn_tensor = input_tensor_dict[fpn_add_list[i]]['data']
#                score = self.conv2d(inputdata=input_tensor, out_channel=64,
#                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))
#                add_tensor = tf.add(deconv,input_fpn_tensor)
#                fused = tf.add(add_tensor, score, name='fuse_{:d}'.format(i + 1))
#                score = fused
#
#            deconv_final = self.deconv2d(inputdata=score, out_channel=64, kernel_size=16,
#                                         stride=8, use_bias=False, name='deconv_final')
#
#            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
#                                      kernel_size=1, use_bias=False, name='score_final')
            
            sad_list = []
            fpn_list = []
            output_list = []
            # for j in range(len(attention_map_list)):
            #     sad_list.append(input_tensor_dict[attention_map_list[j]]['data'])
            # for k in range(len(fpn_concat_list)):
            #     fpn_list.append(input_tensor_dict[fpn_concat_list[k]]['data'])  
            for l in range(len(decode_layer_list)):
                output_list.append(input_tensor_dict[decode_layer_list[l]]['data']) 
            
            deconv_1 = self.deconv2d(inputdata=output_list[0], out_channel=160, kernel_size=3, stride=2, use_bias=False, name='deconv_1')
            
            #fuse_1 = tf.concat([deconv_1, fpn_list[0]], -1)
            
            deconv_2 = self.deconv2d(inputdata=deconv_1, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_2')
            
            #fuse_2 = tf.concat([deconv_2, fpn_list[1]], -1)
            
            # deconv_3 = self.deconv2d(inputdata=deconv_2, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_3')
            
            #fuse_3 = tf.concat([deconv_3, fpn_list[2]], -1)
            
            #deconv_4 = self.deconv2d(inputdata=deconv_3, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_4')
            
            deconv_final = self.deconv2d(inputdata=deconv_2, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_final')
            
            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')
            
            
            ret['logits'] = score_final
            ret['deconv'] = deconv_final
            # ret['output_1'] = sad_list[4]
            # ret['output_2'] = sad_list[3]
            # ret['output_3'] = sad_list[2]
            # ret['output_4'] = sad_list[1]
            # ret['output_5'] = sad_list[0]

            
        #     # score stage 1
        #     input_tensor = input_tensor_dict['pool5']
        #
        #     score_1 = self.conv2d(inputdata=input_tensor, out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_1')
        #
        #     # decode stage 1
        #     deconv_1 = self.deconv2d(inputdata=score_1, out_channel=2, kernel_size=4,
        #                              stride=2, use_bias=False, name='deconv_1')
        #
        #     # score stage 2
        #     score_2 = self.conv2d(inputdata=input_tensor_dict['pool4'], out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_2')
        #
        #     # fuse stage 1
        #     fuse_1 = tf.add(deconv_1, score_2, name='fuse_1')
        #
        #     # decode stage 2
        #     deconv_2 = self.deconv2d(inputdata=fuse_1, out_channel=2, kernel_size=4,
        #                              stride=2, use_bias=False, name='deconv_2')
        #
        #     # score stage 3
        #     score_3 = self.conv2d(inputdata=input_tensor_dict['pool3'], out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_3')
        #
        #     # fuse stage 2
        #     fuse_2 = tf.add(deconv_2, score_3, name='fuse_2')
        #
        #     # decode stage 3
        #     deconv_3 = self.deconv2d(inputdata=fuse_2, out_channel=2, kernel_size=16,
        #                              stride=8, use_bias=False, name='deconv_3')
        #
        #     # score stage 4
        #     score_4 = self.conv2d(inputdata=deconv_3, out_channel=2,
        #                           kernel_size=1, use_bias=False, name='score_4')
        #
        # ret['logits'] = score_4

        return ret


if __name__ == '__main__':

    vgg_encoder = vgg_encoder_swin.VGG16Encoder(phase=tf.constant('train', tf.string))

    decoder = FCNDecoder()

    in_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3],
                               name='input')

    vgg_encode_ret = vgg_encoder.encode(in_tensor, name='vgg_encoder')
    #dense_encode_ret = dense_encoder.encode(in_tensor, name='dense_encoder')
#    decode_ret = decoder.decode(vgg_encode_ret, name='decoder',
#                                decode_layer_list=['pool5','pool4','pool3'] ,
#                                fpn_add_list=['fpn_5','fpn_4'],#,'fpn_3','fpn_2','fpn_1']
#                                attention_map_list=['output_5', 'output_4', 'output_3', 'output_2', 'output_1'])
#    print(decode_ret)
    
