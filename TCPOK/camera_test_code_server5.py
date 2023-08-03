# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:17:38 2020

@author: mediacore
"""

"""
測試LaneNet模型  
"""
from re import A
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
import threading
from PIL import Image

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
            #time.sleep(self.FPS)
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
class a():
    def __init__(self, src=0):
        global stringDatat
        self.lock=threading.Lock
        self.thread = Thread(target=self.recvvv, args=())
        self.thread.daemon = True
        self.thread.start()
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
        

        cluster = lanenet_cluster.LaneNetCluster()
        postprocessor = lanenet_postprocess.LaneNetPoseProcessor()    
        saver = tf.compat.v1.train.Saver()#保存和恢復變量
        
        #控制GPU资源使用率

        sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION#程式最多能佔用指定的視訊記憶體
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH#<------------
        sess_config.gpu_options.allocator_type = 'BFC'#BFC算法

        sess = tf.compat.v1.Session(config=sess_config)

        frame_list = []
        count = 0
        time_predict = 0
        time_cluster = 0
        with sess.as_default():

            saver.restore(sess=sess, save_path=weights_path)
            encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
            while 1:
                #接收
                #msg = 'lcccc'.strip()
                #if len(msg) == 0:
                #    continuesock
                #sock.send(msg.encode('utf-8')
                frame=c()

                if (cv2.waitKey(1)& 0xFF == ord('q')):
                 break
            
                log.info('開始讀取圖像數據並進行預處理')
                t_start = time.time()
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
                binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])     #把有點歪的線刪掉，沒有這行線會比較多
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                            instance_seg_ret=instance_seg_image[0])
                                            
                
            
                # demo image
                #image_merge_temp = np.where((mask_image==0).all(axis=2), image_merge_temp, mask_image)原本ㄉ有bug
                image_merge_temp = np.where((mask_image==0), image_merge_temp, mask_image)
                #cv2.imshow('frame', image_merge_temp)

                #cv2.waitKey(1)
                #image_merge_temp = cv2.resize(image_merge_temp, (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                #                              image_merge_temp       int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                #                                     interpolation=cv2.INTER_LINEAR)
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
                #cv2.imshow('frame',image_merge_temp)s
                #videoWriter.write(image_merge_temp) #寫入影片
                #success, frame = videoCapture.read() #讀取下一幀
                #frame = video_path
                
                #回傳
                result, imgencode = cv2.imencode('.jpg', mask_image, encode_param)
                data = numpy.array(imgencode)
                stringDatat = data.tostring()

                thread2 = b(stringDatat,'thread2')
                thread2.daemon = True
                thread2.start()
            
                #sock.send(str(len(stringData)).ljust(16).encode())
                #sock.send(stringData)
                #decimg=cv2.imdecode(data,1)
                #cv2.imshow('SERVER2',image_merge_temp)
                
                #cv2.waitKey(FPS_MS)
                count += 1
                return  image_merge_temp
                
            print('預測車道線的部分平均每張耗時:', time_predict/count)
            print('進行車道線聚類的部分平均每張耗時:', time_cluster/count)
                
        sess.close()
        cv2.destroyAllWindows()   
class b(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.s = socket.socket()
    def trans(self):
        TCP_IP="192.168.19.1"
        TCP_PORT = 5555
        sock = socket.socket()
        sock.connect((TCP_IP, TCP_PORT))
        while 1:
                sock.send(str(len(stringDatat)).ljust(16).encode())
                sock.send(stringDatat)   
class c():
    def __init__(self):
        Thread.__init__(self)
        self.s = socket.socket()
    def trans():
        TCP_IP="192.168.19.1"
        TCP_PORT = 5555
        sock = socket.socket()
        sock.connect((TCP_IP, TCP_PORT))
        while 1:
            length = a.recvall(sock,16)
            stringData = a.recvall(sock, int(length))
            data = numpy.fromstring(stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
        return decimg 
class c():
    def __init__(self):
        Thread.__init__(self)
        self.s = socket.socket()
    def trans():
        TCP_IP="192.168.19.1"
        TCP_PORT = 5555
        sock = socket.socket()
        sock.connect((TCP_IP, TCP_PORT))
        while 1:
            length = a.recvall(sock,16)
            stringData = a.recvall(sock, int(length))
            data = numpy.fromstring(stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
        return decimg 
def main():
    args = init_args()
    thread2=b()
    thread2.start()
    image_merge_temp=a.test_lanenet(video_path=args.video_path, weights_path=args.weights_path, use_gpu=args.use_gpu, net_flag=args.net_flag)
    cv2.imshow('SERVER2',image_merge_temp)    
    cv2.waitKey(33)


if __name__ == '__main__':
    a()
        
      
    
#https://blog.csdn.net/u014595019/article/details/50178069