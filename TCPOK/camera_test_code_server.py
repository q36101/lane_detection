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

from threading import Thread

from PIL import Image

import threading

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

class VideoStreamWidget(object):
    def __init__(self):
        self.TCP_IP="192.168.19.1"
        self.TCP_PORT = 5555
        self.s = socket.socket()
        self.s.connect((self.TCP_IP, self.TCP_PORT))
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        self.thread1 = Thread(target=self.recvvv, args=())######
        self.thread1.daemon = True
        self.thread1.start()
        #self.stringData=''

        self.length = VideoStreamWidget.recvall(self.s,16)
        self.stringData = VideoStreamWidget.recvall(self.s, int(self.length))
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        
        self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor()    
        self.saver = tf.compat.v1.train.Saver()#保存和恢復變量
        #self.thread = Thread(target=self.update, args=())######
        #self.thread.daemon = True
        #self.thread.start()
        #def update(self):
    def recvall(sock, coun):
        buf = b''
        while coun:
            newbuf = sock.recv(coun)
            if not newbuf: return None
            buf += newbuf
            coun -= len(newbuf)
        return buf
    def show_frame(self):

        if self.stringData !='':
            data = numpy.fromstring(self.stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)

            input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3], name='input_tensor')#为将始终输入的张量插入占位符
            phase_tensor = tf.constant('train', tf.string)

            net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='mv3')
            binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')

            sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})

            sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION#程式最多能佔用指定的視訊記憶體
            sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
            sess_config.gpu_options.allocator_type = 'BFC'

            sess = tf.compat.v1.Session(config=sess_config)
            
            frame_list = []
            count = 0
            time_predict = 0
            time_cluster = 0

            self.saver.restore(sess=sess, save_path='D:/Users/mediacore/lane_detection/model/culane_lanenet/culane_lanenet_mv3_2022-05-16-16-09-10.ckpt-288000')
            #print(success, 'processsssssssssssssssssssssssssssssssssssss')

            frame=decimg

            t_start = time.time()          
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

            log.info('圖像讀取完畢, 耗時: {:.5f}s'.format(time.time() - t_start))#將浮點數舍入到小數點後5位
            
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
            binary_seg_image[0] = self.postprocessor.postprocess(binary_seg_image[0])     #把有點歪的線刪掉，沒有這行線會比較多
            mask_image = self.cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])
                                           
            image_merge_temp = np.where((mask_image==0), image_merge_temp, mask_image)

            t_post_process_cost = time.time() - t_start
            log.info('單張圖像車道線聚類耗時: {:.5f}s'.format(t_post_process_cost))
            if(count == 0):
                time_cluster = time_cluster
            else:
                time_cluster = time_cluster + t_post_process_cost
            
            #回傳
            result, imgencode = cv2.imencode('.jpg', mask_image, encode_param)
            data = numpy.array(imgencode)
            stringData = data.tostring()
        
            self.s.send(str(len(stringData)).ljust(16).encode());
            self.s.send(stringData);
        
            #decimg=cv2.imdecode(data,1)
            cv2.imshow('SERVER2',image_merge_temp)
            
            cv2.waitKey(1)
            count += 1
            
            
        print('預測車道線的部分平均每張耗時:', time_predict/count)
        print('進行車道線聚類的部分平均每張耗時:', time_cluster/count)

    def recvvv(self):
        i=0
        
        while True:
            self.buf = b''
            self.count=16
            while self.count:
                self.newbuf =self.s.recv(self.count)
                if not self.newbuf: return None
                self.buf += self.newbuf
                self.count -= len(self.newbuf)
                #print('A',self.count ,len(self.newbuf))
            self.length=self.buf
            self.buf = b''
            self.count=int(self.length)
            #print('B',self.count)
            while int(self.count):
                #print('B',self.length)
                self.newbuf=self.s.recv(self.count)
                if not self.newbuf: return None
                self.buf += self.newbuf
                self.count -= len(self.newbuf)
                #print('B',int(self.length))
            self.stringData=self.buf

#TCP_IP = "169.254.14.65"

'''
TCP_IP = "127.0.0.1"
TCP_PORT = 5555
'''
#TCP_IP='localhost'
#TCP_PORT = 5555
#s = socket.socket()
#s.connect((TCP_IP, TCP_PORT))
#encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
video_stream_widget = VideoStreamWidget()
while 1:
    video_stream_widget.show_frame()
    #msg = 'lcccc'.strip()
    #if len(msg) == 0:
    #    continue
    #sock.send(msg.encode('utf-8'))
    
    #length = recvall(sock,16)
    #stringData = recvall(sock, int(length))
    #data = numpy.fromstring(stringData, dtype='uint8')
    #decimg=cv2.imdecode(data,1)
    #cv2.imshow('CLIENT2',decimg)
    #cv2.waitKey(1)

    
sock.close()
cv2.destroyAllWindows()



    


        
      
        

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

