# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:17:38 2020

@author: mediacore
"""

"""
測試LaneNet模型  
"""
import sys
sys.path.append('/home/mediacore/lane_detection') #path to fpn-lane-detection
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='The vidoe path')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--net_flag', type=str, help='The model you use to train')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()

def test_lanenet(video_path, weights_path, net_flag, use_gpu):

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3], name='input_tensor')
    phase_tensor = tf.constant('train', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag=net_flag)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')

    videoCapture = cv2.VideoCapture(video_path) #影像擷取frame

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

    videoWriter = cv2.VideoWriter('LaneNet.avi', fourcc, fps, size)
    
    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()    
    saver = tf.compat.v1.train.Saver()
    
    if use_gpu:
        sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
        print('You are using GPU!!!!!!!!!!!!!!!!')
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        print('You are not using GPU!!!!!!!!!!!!!!!!')
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)
    
    frame_list = []
    count = 0
    time_predict = 0
    time_cluster = 0

    success, frame = videoCapture.read()

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        print(success, 'processsssssssssssssssssssssssssssssssssssss')

        # Set sess configuration
        while success:

            log.info('開始讀取圖像數據並進行預處理')
            t_start = time.time()
            #image = frame
        
            #T=2適用，若T有改，這裡要再改        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if count == 0:
                frame_list.append(frame)
                frame_list.append(frame)
                image_list = [cv2.resize(tmp, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR) for tmp in frame_list]
                image_merge_temp = image_list[1]
                image_list = [tmp - VGG_MEAN for tmp in image_list]
        
            else:
                frame_list.append(frame)
                frame_list_new = frame_list[count:count+2]
                image_list = [cv2.resize(tmp, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR) for tmp in frame_list_new]
                image_merge_temp = image_list[1]
                image_list = [tmp - VGG_MEAN for tmp in image_list]

            log.info('圖像讀取完畢, 耗時: {:.5f}s'.format(time.time() - t_start))
            
            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: image_list})
            t_detection_cost = time.time() - t_start
            log.info('單張圖像車道線預測耗時: {:.5f}s'.format(t_detection_cost))
            if(count == 0):
                time_predict = time_predict
            else:
                time_predict = time_predict + t_detection_cost

            t_start = time.time()
            binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])     #把有點歪的線刪掉，沒有這行線會比較多
            mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])
            
            # demo image
            #image_merge_temp = np.where((mask_image==0).all(axis=2), image_merge_temp, mask_image)原本ㄉ有bug
            image_merge_temp = np.where((mask_image==0), image_merge_temp, mask_image)
            image_merge_temp = cv2.resize(image_merge_temp, (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                                 interpolation=cv2.INTER_LINEAR)
            t_post_process_cost = time.time() - t_start
            log.info('單張圖像車道線聚類耗時: {:.5f}s'.format(t_post_process_cost))
            if(count == 0):
                time_cluster = time_cluster
            else:
                time_cluster = time_cluster + t_post_process_cost
            #mask image
            #mask_image = cv2.resize(mask_image, (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 #int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                                 #interpolation=cv2.INTER_LINEAR)
            #videoWriter.write(mask_image) #寫入影片
            videoWriter.write(image_merge_temp) #寫入影片
            success, frame = videoCapture.read() #讀取下一幀
            count += 1
        print('預測車道線的部分平均每張耗時:', time_predict/count)
        print('進行車道線聚類的部分平均每張耗時:', time_cluster/count)
    sess.close()


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(video_path=args.video_path, weights_path=args.weights_path, use_gpu=args.use_gpu, net_flag=args.net_flag)

