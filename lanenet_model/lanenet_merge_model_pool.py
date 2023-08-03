# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:22:24 2020

@author: mediacore
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
實現LaneNet模型
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

#from encoder_decoder_model import Inception_ResNet_v2_encoder
#from encoder_decoder_model import Inception_ResNet_v2_decoder
from encoder_decoder_model import vgg_encoder_swin
from encoder_decoder_model import fcn_decoder_swin
from encoder_decoder_model import cnn_basenet_swin
#from encoder_decoder_model import MobileNet_v2_encoder
#from encoder_decoder_model import MobileNet_v2_decoder
#from encoder_decoder_model import ERFNet_encoder
#from encoder_decoder_model import ERFNet_decoder
#from encoder_decoder_model import resnet_encoder
#from encoder_decoder_model import resnet_decoder
# from encoder_decoder_model import mobilenet_v3_large_encoder_swintf
# from encoder_decoder_model import mobilenet_v3_large_encoder_udlr_susd as mobilenet_v3_large_encoder_swintf
# from encoder_decoder_model import mobilenet_v3_large_encoder_add_then_conv as mobilenet_v3_large_encoder_swintf
from encoder_decoder_model import mobilenet_v3_large_encoder_pool 
from encoder_decoder_model import mobilenet_v3_large_decoder_pool
from lanenet_model import lanenet_discriminative_loss_swin




class LaneNet(cnn_basenet_swin.CNNBaseModel):
    """
    實現語義分割模型
    """
    def __init__(self, phase, net_flag): #net_flag要記得改!!!!!!!!!!!!!!!!!!!!!!! 指定vgg只是方便run這個py檔
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder_swin.VGG16Encoder(phase=phase)
            self._decoder = fcn_decoder_swin.FCNDecoder()
            
        # elif self._net_flag == 'irv2':
        #     self._encoder = Inception_ResNet_v2_encoder.Inception_ResNet_v2_Encoder(phase=phase)
        #     self._decoder = Inception_ResNet_v2_decoder.Inception_ResNet_v2_Decoder()
            
#         elif self._net_flag == 'v2':
#             self._encoder = MobileNet_v2_encoder.MobileNet_v2_Encoder(phase=phase)
#             self._decoder = MobileNet_v2_decoder.MobileNet_v2_Decoder()
            
#         elif self._net_flag == 'erf':
#             self._encoder = ERFNet_encoder.ERFNet_Encoder(phase=phase)
#             self._decoder = ERFNet_decoder.ERFNet_Decoder(phase=phase)
            
#         elif self._net_flag == 'res':
#             self._encoder = resnet_encoder.Resnet_Encoder(phase=phase)
#             self._decoder = resnet_decoder.Resnet_Decoder()
            
        elif self._net_flag == 'mv3':
            self._encoder = mobilenet_v3_large_encoder_pool.mobilenet_v3_large_Encoder(phase=phase)
            self._decoder = mobilenet_v3_large_decoder_pool.mobilenet_v3_large_Decoder()
        
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor, name):
        """
        前向傳播過程
        :param input_tensor:
        :param name:
        :return:
        """
        # input_tensor=tf.image.random_brightness(input_tensor,0.5)
        # input_tensor=tf.image.random_contrast(input_tensor,lower=0.8,upper=3)
        # input_tensor=tf.image.random_hue(input_tensor,max_delta=0.2)
        # input_tensor=tf.image.random_saturation(input_tensor,lower=0,upper=2)
        with tf.compat.v1.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',                                                 
                                                  decode_layer_list=['processed_feature'])
                                                  # fpn_concat_list=['conv_5_3','conv_4_3','conv_3_3'],
                                                  # attention_map_list=['output_5','output_4',                                                                      
                                                  #                     'output_3','output_2','output_1'])
                return decode_ret
            
            elif self._net_flag.lower() == 'v2':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['conv_9_1'])    
                return decode_ret  

            elif self._net_flag.lower() == 'irv2':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dropout_output'])    
                return decode_ret
            
            elif self._net_flag.lower() == 'erf':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['layer_16'],
                                                  fpn_add_list=['fpn_15', 'fpn_13','fpn_11', 'fpn_9'],
                                                  attention_map_list=['output_16','output_15','output_14', 'output_13','output_12','output_11',
                                                                      'output_10','output_9','output_8', 'output_7','output_2'])
                return decode_ret
            
            elif self._net_flag.lower() == 'res':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',  
                                                  decode_layer_list=['processed_feature'],
                                                  #decode_layer_list=['conv_panet_4'],
                                                  fpn_concat_list=['conv_4_6', 'conv_3_4', 'conv_2_3'],
                                                  #fpn_add_list=['conv_panet_3', 'conv_panet_2', 'conv_panet_1'],
                                                  attention_map_list=['output_5','output_4',
                                                                      'output_3','output_2','output_1'])
                return decode_ret
            
            elif self._net_flag.lower() == 'mv3':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',                                            
                                                  #decode_layer_list=['processed_feature'],
                                                  decode_layer_list=['result'],
                                                  fpn_concat_list=['bneck6_2', 'bneck4_3', 'bneck3_2','bneck2_1'],    #拿掉最前面'bneck6_2'，加'bneck2_1'
                                                  attention_map_list=['output_5','output_4','output_3','output_2','output_1'])
                return decode_ret 

    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        計算LaneNet模型損失函數
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.compat.v1.variable_scope(name):
            # 前向傳播獲取logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')      #會得到decoder完的結果
            
            # 計算SAD損失函數                 
#            sad_loss = tf.reduce_sum((inference_ret['output_2']-inference_ret['output_1'])**2)/2
#            sad_loss = tf.add(tf.reduce_sum((inference_ret['output_3']-inference_ret['output_2'])**2)/2, sad_loss)
#            sad_loss = tf.add(tf.reduce_sum((inference_ret['output_6']-inference_ret['output_5'])**2)/2, sad_loss)
#            sad_loss = tf.add(tf.reduce_sum((inference_ret['output_4']-inference_ret['output_3'])**2)/2, sad_loss)
            #print(sad_loss, 'GGGGGGGGGGGGG') = Tensor("loss/Add_2:0", shape=(), dtype=float32)
                            
            # 計算二值分割損失函數
            decode_logits = inference_ret['logits']       
            
            decode_logits_sum = tf.expand_dims(tf.reduce_sum(tf.abs(decode_logits)**2, axis=3), axis=3)
            #print(inference_ret)# = ('logits':(1, 256, 512, 2), 'deconv':(1, 256, 512, 64))
            
            decode_logits_reshape = tf.reshape(           #這個數值每個epoch都不同(正常)
                decode_logits,
                shape=[decode_logits.get_shape().as_list()[0],
                       decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
                       decode_logits.get_shape().as_list()[3]])    
            #print(decode_logits_reshape)# = (1, 131072, 2)      #decode完後，每個點所代表哪一個label(背景OR線)，故會有2個channel
            
            binary_label_for_dice = binary_label
            
            binary_label = tf.compat.v1.squeeze(binary_label, squeeze_dims=[3])           #把binary_label的第三個值去掉
            binary_label_for_focal = tf.one_hot(binary_label, depth=2)
            #print(binary_label)# = (1, 256, 512)
            
            binary_label_reshape = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0],
                       binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
            #print(binary_label_reshape)# = (1, 131072)
            
            binary_label_reshape = tf.one_hot(binary_label_reshape, depth=2)             #這個數值每個epoch都相同(都是131072.00000)
            #print(binary_label_reshape)# = (1, 131072, 2)
            
            binary_difference = binary_label_reshape - decode_logits_reshape
            l2_loss = tf.nn.l2_loss(binary_difference)  ### L2-loss ##############################################
            l2_loss = tf.reduce_mean(l2_loss) 
            
            # binary_segmenatation_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     logits=decode_logits, labels=tf.squeeze(binary_label, squeeze_dims=[3]),
            #     name='entropy_loss')
            binary_segmenatation_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                labels=binary_label_reshape, logits=decode_logits_reshape, name='entropy_loss')      #算cross entropy loss 
            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)
#TEST            binary_segmenatation_loss = tf.nn.l2_loss(decode_logits_reshape)
            #print(binary_segmenatation_loss, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') = Tensor("loss/Mean:0", shape=(), dtype=float32)
            
            # 計算discriminative loss損失函數
            decode_deconv = inference_ret['deconv']      # = (1, 256, 512, 64)
            # 像素嵌入
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=3, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')   # = (1, 256, 512, 3)
            #計算discriminative loss
            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            disc_loss, l_var, l_dist, l_reg = \
                lanenet_discriminative_loss_swin.discriminative_loss(
                    pix_embedding, instance_label, 3, image_shape, 0.5, 1.5, 1.0, 1.0, 0.001)        #算實例分割的loss(disc_loss是裡面的total_loss) !!!!!!!1.5可以改3試試!!
                
            #Dice loss
            TP = tf.compat.v1.count_nonzero(tf.multiply(tf.cast(tf.equal(binary_label_for_dice, 1), tf.int32),     #將預測的圖和gt做相乘，算交集(PIXEL值為1)的點的個數
                                                  tf.cast(tf.equal(decode_logits, 1), tf.int32)),         #tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False
                                      dtype=tf.int32)                                                   #再用tf.cast轉int32，會把True轉為1，False轉為0
            union_1 = tf.add(tf.compat.v1.count_nonzero(tf.cast(tf.equal(binary_label_for_dice, 1),        #將預測為線的PIXEL數和GT為線的PIXEL數相加，算聯集
                                              tf.int32), dtype=tf.int32),
                             tf.compat.v1.count_nonzero(tf.cast(tf.equal(decode_logits, 1),
                                              tf.int32), dtype=tf.int32))
            union_1 = tf.subtract(union_1, TP)   #因為剛剛把交集部分多加一遍，這邊把多加的減掉
            IoU = tf.divide(tf.cast(TP, tf.float32), tf.cast(union_1, tf.float32))
            IoU = tf.reduce_mean(IoU)
            #print('888888888888888888888888888888888888888888888888888888888888')
            #print(TP)
            #print(IoU)
    
            #FN = tf.count_nonzero(tf.cast(tf.equal((binary_label_for_dice - decode_logits), 1), tf.int32), dtype=tf.int32)    #GT有，但沒偵測到
            #FP = tf.count_nonzero(tf.cast(tf.equal((decode_logits - binary_label_for_dice), 1), tf.int32), dtype=tf.int32)    #GT沒有，但有偵測到(亂偵測)
    
            dice_loss = tf.divide((2*tf.cast(TP, tf.float32) + 1), (tf.cast(union_1, tf.float32) + 1))
            dice_loss = 1 - tf.reduce_mean(dice_loss)
            
            #focal loss
            softmax_prediction = tf.nn.softmax(decode_logits)
            softmax_prediction = tf.reduce_mean(softmax_prediction*tf.cast(binary_label_for_dice, tf.float32))
            logpt = tf.compat.v1.log(softmax_prediction)

            loss = -0.25 * (1-softmax_prediction)**2 * logpt
            focal_loss = tf.reduce_mean(loss)
            


            # 合併損失
            total_loss = binary_segmenatation_loss + 0.3 * disc_loss #+ 0.001 * l2_loss + focal_loss# + dice_loss + 0.001 * l2_loss# + focal_loss #+ 0.1 * sad_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': disc_loss,
                'l2_loss': l2_loss,
                'dice_loss': dice_loss,
                'focal_loss': focal_loss,
#                'sad_loss': sad_loss,
                'output_1': inference_ret['output_1'],
                'output_2': inference_ret['output_2'],
                'output_3': inference_ret['output_3'],
                'output_4': inference_ret['output_4'],
                'output_5': inference_ret['output_5'],
                # 'output_x': inference_ret['output_x'],
                # 'output_y': inference_ret['output_y'],
#                'output_6': inference_ret['output_6'],
                'decode_binary': decode_logits_sum
            }

            return ret

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.compat.v1.variable_scope(name):
            # 前向傳播獲取logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')          #會得到decoder完的結果
            # 計算二值分割損失函數
            decode_logits = inference_ret['logits']      #('logits':(1, 256, 512, 2), 'deconv':(1, 256, 512, 64))
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)      #取得每一個list裡的最大值的index(在第幾個位置)
            # 計算像素嵌入
            decode_deconv = inference_ret['deconv']
            # 像素嵌入
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=3, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')       # = (1, 256, 512, 3)
            # pix_embedding = binary_seg_ret

            return binary_seg_ret, pix_embedding


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    print(ret['total_loss'])
