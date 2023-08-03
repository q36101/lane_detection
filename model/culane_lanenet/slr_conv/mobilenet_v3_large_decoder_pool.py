# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:23:26 2019

@author: mediacore
"""

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
from tensorflow.keras import layers
from encoder_decoder_model import cnn_basenet_swin
from encoder_decoder_model import mobilenet_v3_large_encoder_swintf
# from encoder_decoder_model import mobilenet_v3_large_encoder_udlr_susd as mobilenet_v3_large_encoder_swintf

class mobilenet_v3_large_Decoder(cnn_basenet_swin.CNNBaseModel):
    """
    實現一個全卷積解碼類
    """
    def __init__(self):
        """

        """
        super(mobilenet_v3_large_Decoder, self).__init__()

    def decode(self, input_tensor_dict, decode_layer_list, name, attention_map_list, fpn_concat_list):
        """
        解碼特徵信息反捲積還原
        :param input_tensor_dict:
        :param decode_layer_list: 需要解碼的層名稱需要由深到淺順序寫
                                  eg. ['processed_feature']
        :param fpn_concat_list: ['bneck6_2', 'bneck4_3', 'bneck3_2']   
        :param attention_map_list: [output_4', 'output_3', 'output_2', 'output_1']                        
        :param name:
        :return:
        """
        ret = dict()

        with tf.compat.v1.variable_scope(name):
            

            sad_list = []
            fpn_list = []
            output_list = []
            for j in range(len(attention_map_list)):
                sad_list.append(input_tensor_dict[attention_map_list[j]]['data'])
            for k in range(len(fpn_concat_list)):
                fpn_list.append(input_tensor_dict[fpn_concat_list[k]]['data'])  
            for l in range(len(decode_layer_list)):
                output_list.append(input_tensor_dict[decode_layer_list[l]]['data']) 
            
            deconv_1 = self.deconv2d(inputdata=output_list[0], out_channel=160, kernel_size=3, stride=2, use_bias=False, name='deconv_1')  #[2, 16, 32, 160]
            
            fuse_1 = tf.concat([deconv_1, fpn_list[0]], -1)  #[2, 16, 32, 272]
            # fuse_1 = layers.concatenate([deconv_1, fpn_list[0]], axis=-1)
            # fuse_1 = tf.compat.v1.reshape(fuse_1, (int(fuse_1.shape[0]), int(fuse_1.shape[1]),int(fuse_1.shape[2]),int(fuse_1.shape[3]))) 
            print('deconv_1',deconv_1)
            print('fuse_1',fuse_1,type(fuse_1))
            print(fuse_1.shape[3],type(fuse_1.shape[3]))
            
            fuse_1 = self.conv2d(inputdata=fuse_1, out_channel=160, kernel_size=1, use_bias=False, name='fuse_1')
            sigmoid_1 = tf.math.sigmoid(fuse_1)
            pixel_product_1 = tf.multiply(sigmoid_1, deconv_1)  #點對點相乘
            result_1 = tf.concat([pixel_product_1, fpn_list[0]], -1)  #[2, 16, 32, 272]
            
            deconv_2 = self.deconv2d(inputdata=result_1, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_2')  #[2, 32, 64, 64]
            
            fuse_2 = tf.concat([deconv_2, fpn_list[1]], -1)  #[2, 32, 64, 104]
            
            fuse_2 = self.conv2d(inputdata=fuse_2, out_channel=64, kernel_size=1, use_bias=False, name='fuse_2')
            sigmoid_2 = tf.math.sigmoid(fuse_2)
            pixel_product_2 = tf.multiply(sigmoid_2, deconv_2)  #點對點相乘
            result_2 = tf.concat([pixel_product_2, fpn_list[1]], -1)  #[2, 16, 32, 104]
            
            deconv_3 = self.deconv2d(inputdata=result_2, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_3')  #[2, 64, 128, 64]
            
            fuse_3 = tf.concat([deconv_3, fpn_list[2]], -1)  #[2, 64, 128, 88]
            
            fuse_3 = self.conv2d(inputdata=fuse_3, out_channel=64, kernel_size=1, use_bias=False, name='fuse_3')
            sigmoid_3 = tf.math.sigmoid(fuse_3)
            pixel_product_3 = tf.multiply(sigmoid_3, deconv_3)  #點對點相乘
            result_3 = tf.concat([pixel_product_3, fpn_list[2]], -1)  #[2, 16, 32, 88]
            
            deconv_4 = self.deconv2d(inputdata=result_3, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_4')  #[2, 128, 256, 64]
            
            deconv_final = self.deconv2d(inputdata=deconv_4, out_channel=64, kernel_size=3, stride=2, use_bias=False, name='deconv_final')  #[2, 256, 512, 64]
            
            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')  #[2, 256, 512, 2]
            
            output_d1 = tf.image.resize(tf.expand_dims(tf.reduce_mean(tf.abs(deconv_1)**2, axis=3), axis=3), (128, 256))
            output_d1 = tf.compat.v1.layers.batch_normalization(output_d1)
            
            output_r1 = tf.image.resize(tf.expand_dims(tf.reduce_mean(tf.abs(result_1)**2, axis=3), axis=3), (128, 256))
            output_r1 = tf.compat.v1.layers.batch_normalization(output_r1)

            ret['logits'] = score_final
            ret['deconv'] = deconv_final
            # ret['output_y'] = sad_list[6]
            # ret['output_x'] = sad_list[5]
            ret['output_1'] = sad_list[4]
            ret['output_2'] = sad_list[3]
            ret['output_3'] = sad_list[2]
            ret['output_4'] = sad_list[1]
            ret['output_5'] = sad_list[0]
            ret['output_d1'] = output_d1
            ret['output_r1'] = output_r1

            

        return ret


if __name__ == '__main__':

    mobilenet_v3_large_encoder = mobilenet_v3_large_encoder_swintf.mobilenet_v3_large_Encoder(phase=tf.constant('train', tf.string))
    
    decoder = mobilenet_v3_large_Decoder()

    in_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3],
                               name='input')

    mobilenet_v3_large_encoder_ret = mobilenet_v3_large_encoder.encode(in_tensor, name='mobilenet_v3_large_encoder')
    decode_ret = decoder.decode(mobilenet_v3_large_encoder_ret, name='decoder',
                                decode_layer_list=['conv_9_1'])
                              
    
