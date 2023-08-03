# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:31:51 2022

@author: Lee
"""

import socket
import cv2
import numpy
from PIL import Image
import os
import torch
import torch.utils.data
import pandas as pd
import numpy as np
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from model.model_vtgnet import FeatExtractor, TrajGenerator
import cv2
from scipy import interpolate
from canlib import canlib, Frame
import time
result=10
import cv2
import os 
import pandas as pd
from threading import Thread
import math
'''
ch = canlib.openChannel(
    channel=0,
    flags=canlib.Open.EXCLUSIVE,
    bitrate= canlib.Bitrate.BITRATE_500K,
)
'''


#interval_before = 11 # 1.5 s
#interval_after = 22 # 3 s
#feature_size = 512
from canlib import canlib, Frame
#torch.cuda.set_device(0)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model1_s = FeatExtractor(feature_size=feature_size).to(device) # keep straight
#model2_s = TrajGenerator(feature_size=feature_size).to(device)
#model_path = './model2/weights/'
#model1_s.load_state_dict(torch.load(model_path + '0-model1.pth', map_location=lambda storage, loc: storage))
#model2_s.load_state_dict(torch.load(model_path + '0-model2.pth', map_location=lambda storage, loc: storage))
#model1_s.eval()
#model2_s.eval()
#csv_path = 'VTG-Driving-Dataset/dataset_straight3.csv'
#data1 = pd.read_csv(csv_path)
#piii=0
#features = torch.Tensor(1, 12, feature_size).to(device)
class VideoStreamWidget(object):
    def __init__(self):
        self.interval_before = 11
        self.interval_after = 22
        self.feature_size = 512
        torch.cuda.set_device(0)
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model1_s = FeatExtractor(feature_size=self.feature_size).to(self.device) # keep straight
        self.model2_s = TrajGenerator(feature_size=self.feature_size).to(self.device)
        self.model1_s.eval()
        self.model2_s.eval()
        self.csv_path = 'VTG-Driving-Dataset/dataset_straight3.csv'
        self.csvdata1 = pd.read_csv(self.csv_path)
        self.piii=0
        self.features = torch.Tensor(1, 12, self.feature_size).to(self.device)
        self.TCP_IP='localhost'
        self.TCP_PORT = 5555
        self.s = socket.socket()
        self.s.connect((self.TCP_IP, self.TCP_PORT))
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        self.thread1 = Thread(target=self.recvvv, args=())######
        self.thread1.daemon = True
        self.thread1.start()
        self.stringData=''
        #self.thread = Thread(target=self.update, args=())######
        #self.thread.daemon = True
        #self.thread.start()
        #def update(self):
    def show_frame(self):

        if self.stringData !='':
            data = numpy.fromstring(self.stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
            #print(data.shape)
            #print("-----------")
            #print(decimg)
            cv2.imshow('CLIENT22222',decimg)
            cv2.waitKey(1)
            with torch.no_grad():
                idx=1
                info_st_index = 3 + self.interval_after + self.interval_before + 1 
                info_st_index_2 = (info_st_index + 3*(self.interval_before+1))
                info_history = self.csvdata1.iloc[idx, info_st_index:info_st_index_2].to_numpy().reshape(-1, 3)
                info_history_net = info_history
                info_history_net = torch.from_numpy(info_history_net.astype('float')).unsqueeze(0).to(self.device)
                info_future = self.csvdata1.iloc[idx, info_st_index_2:].to_numpy().reshape(-1,3)
                local_x_history = info_history[:,0]
                local_y_history = info_history[:,1]
                spd_history = info_history[:,2]
                image = []
                if self.piii<12:
                    image=Image.fromarray(cv2.cvtColor(decimg,cv2.COLOR_BGR2RGB))
                    image = torch.stack([transforms.ToTensor()(transforms.Resize((224, 224))(image))], dim=0)
                    image[0, :, :, :] = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(image[0, :, :, :]) 
                    image = image.unsqueeze(0).to(self.device)
                    self.features[:, self.piii, :] = self.model1_s(image[:, 0, :, :, :])
                    self.piii=self.piii+1
                else:
                    image=Image.fromarray(cv2.cvtColor(decimg,cv2.COLOR_BGR2RGB))
                    image = torch.stack([transforms.ToTensor()(transforms.Resize((224, 224))(image))], dim=0)
                    image[0, :, :, :] = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(image[0, :, :, :])
                    image = image.unsqueeze(0).to(self.device)
                    self.features[:, :11, :] =self.features[:, 1:, :].clone()
                    self.features[:, 11, :] = self.model1_s(image[:, 0, :, :, :])
                    outputs, logvar, attentions = self.model2_s(self.features, info_history_net)
                    planned = (outputs.reshape(-1,3).cpu().detach().numpy())
                    local_x_planned = planned[:,0]
                    local_y_planned = planned[:,1]
                    spd_planned = planned[:,2]
                    print('spd_planned ',spd_planned )
                    
                    self.s.send( str(len(self.stringData)).ljust(16).encode());
                    self.s.send( self.stringData );


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
