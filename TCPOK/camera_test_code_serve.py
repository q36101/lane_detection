# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:17:38 2020

@author: mediacore
"""

"""
測試LaneNet模型  
"""
import sys
from tkinter import W
sys.path.append('/home/mediacore/lane_detection') #path to fpn-lane-detection

import os
import os.path as ops
import argparse
import time
import math
import socket
import numpy

import tensorflow as tf
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
tf.compat.v1.disable_eager_execution()
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

from PIL import Image

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()#先創造 argparse.ArgumentParser() 物件，它是我們管理引數需要的 “parser”
    
    parser.add_argument('--video_path', type=str, help='The vidoe path')#add_argument() 告訴 parser 我們需要的命令列引數有哪些
    parser.add_argument('--weights_path', type=str, help='The model weights path')#type=str:要求使用者輸入引數「必須」是 str 型別
    parser.add_argument('--net_flag', type=str, help='The model you use to train')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)
    parser.add_argument('--use_cam', '-c', help='If use this parameter means use camera as input instead of mp4', action='store_true', default=False)

    return parser.parse_args()#使用 parse_args() 來從 parser 取得引數傳來的 Data
def recvall(sock, coun):
    buf = b''
    while coun:
        newbuf = sock.recv(coun)
        if not newbuf: return None
        buf += newbuf
        coun -= len(newbuf)
    return buf


def test_lanenet(video_path, weights_path, net_flag, use_gpu):

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3], name='input_tensor')#为将始终输入的张量插入占位符
    #dtype：张量中要输入的元素的类型。
    #shape：要输入的张量的形状（可选）。 如果未指定形状，则可以输入任何形状的张量。
    #name：操作的名称（可选）
    phase_tensor = tf.constant('train', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag=net_flag)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')
    
    
    #if(use_cam):
    #    videoCapture = cv2.VideoCapture(0)
    #else:
    #   videoCapture = cv2.VideoCapture(video_path) #影像擷取frame
    #print('000000000000000000000000000000000000000000000000000000000000')
    #print(type(video_path))
    #print(type(videoCapture))

    #fps = videoCapture.get(cv2.CAP_PROP_FPS)
    #size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    
    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()    
    saver = tf.compat.v1.train.Saver()#保存和恢復變量
    
    #控制GPU资源使用率
    if use_gpu:
        # sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})#参数 device_count，使用字典分配可用的 GPU 设备号和 CPU 设备号。
        sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})#参数 device_count，使用字典分配可用的 GPU 设备号和 CPU 设备号。
        print('You are using GPU!!!!!!!!!!!!!!!!')
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        print('You are not using GPU!!!!!!!!!!!!!!!!')
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION#程式最多能佔用指定的視訊記憶體
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH#<------------
    sess_config.gpu_options.allocator_type = 'BFC'#BFC算法

    sess = tf.compat.v1.Session(config=sess_config)
    #TCP_IP = "192.168.43.218"
    #TCP_IP="110.110.110.11"
    TCP_IP="localhost"
    TCP_PORT = 5555

    sock = socket.socket()
    sock.connect((TCP_IP, TCP_PORT))
    
    frame_list = []
    count = 0
    i=1
    #frame =np.zeros((256, 512, 3), dtype='uint8')*0
    #frame = video_path #從攝影機擷取一張影像
    #success, frame = videoCapture.read()
    #cv2.imshow('frame', frame)
    print('11111111111111111111111111111111111111111111111')
    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        #print(success, 'processsssssssssssssssssssssssssssssssssssss')
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        # Set sess configuration
        while 1:
            #接收
            #msg = 'lcccc'.strip()
            #if len(msg) == 0:
            #    continuesock
            #sock.send(msg.encode('utf-8'))
            length = recvall(sock,16)
            stringData = recvall(sock, int(length))
            data = numpy.fromstring(stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
            t_start1 = time.time()
            frame = decimg
            if i<50:
                a=10
            else:
                a=4
            if i%a==1:  
                
                t_start = time.time()
                print('00000000000000000000',time.time() - t_start)
                t_start = time.time()
                if (cv2.waitKey(1)& 0xFF == ord('q')):
                    break
                log.info('開始讀取圖像數據並進行預處理')
                #cv2.imshow('frame',frame)
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
                count += 1

                t_start = time.time()

                binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: image_list})
                print('first',time.time() - t_start)
                t_start = time.time()
                '''
                if(count == 0):
                    time_predict = time_predict
                else:
                    time_predict = time_predict + t_detection_cost
                '''
                binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])     #把有點歪的線刪掉，沒有這行線會比較多
                print('0111111111',time.time() - t_start)
                t_start = time.time()
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                            instance_seg_ret=instance_seg_image[0])
                image_merge_temp = np.where((mask_image==0), image_merge_temp, mask_image)
                print('second',time.time() - t_start)
                mask_image = cv2.resize(mask_image,(640,480),interpolation=cv2.INTER_LINEAR)
                t_start = time.time()
                #print(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
                #print(int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                
                # demo image
                #image_merge_temp = np.where((mask_image==0).all(axis=2), image_merge_temp, mask_image)原本ㄉ有bug
                
                
                result, imgencode = cv2.imencode('.jpg', mask_image, encode_param)
                data = numpy.array(imgencode)
                stringData = data.tostring()
            
                sock.send(str(len(stringData)).ljust(16).encode());
                sock.send(stringData)
                #cv2.imshow('frame', image_merge_temp)
                #print(image_merge_temp.shape)
                frame = decimg
            else:
                t_start = time.time()
                frame = decimg
                print('0111111111',time.time() - t_start)
            #success, frame = videoCapture.read() #讀取下一幀
            i+=1
            print('i',i)
            print('count',count)
            print('77777777777777777777777777777777777777777',time.time() - t_start1)

if __name__ == '__main__':
    # init args
    args = init_args()

while 1:

    test_lanenet(video_path=args.video_path, weights_path=args.weights_path, use_gpu=args.use_gpu, net_flag=args.net_flag)

        
      
        

# 選擇第二隻攝影機
#cap = cv2.VideoCapture(1)

#while(True):
  # 從攝影機擷取一張影像
  #ret, frame = cap.read()

  # 顯示圖片
  #cv2.imshow('frame', frame)

  # 若按下 q 鍵則離開迴圈
  #if cv2.waitKey(1) & 0xFF == ord('q'):
   # break

# 釋放攝影機
#cap.release()

# 關閉所有 OpenCV 視窗
#cv2.destroyAllWindows()
#mport cv2
'''
                if  i<10:
                    image_merge_temp=np.zeros((256, 512, 3), dtype='uint8')*0
                    mask_image=np.zeros((256, 512, 3), dtype='uint8')*0
                else:
                    image_merge_temp=image_merge_temp
                    mask_image=mask_image
'''
