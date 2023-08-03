# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:31:51 2022

@author: Lee
"""
from threading import Thread
import socket
import cv2, time
import numpy
#import pandas as pd
import numpy as np
from scipy import interpolate




class VideoStreamWidget(object):
    def __init__(self, src=0):
        
        self.TCP_IP = ""
        self.TCP_PORT = 5555
        self.sp=0
        self.stringData=''
        self.stringData1=''
        self.stringData2=''
        self.stringData3=''
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.s.bind((self.TCP_IP, self.TCP_PORT))
        self.s.listen(True)
        self.conn, self.addr = self.s.accept()
        self.conn1, self.addr1 = self.s.accept()
        self.conn2, self.addr2 = self.s.accept()
        self.conn3, self.addr3 = self.s.accept()
        #self.capture = cv2.VideoCapture(src)
        self.capture = cv2.VideoCapture('C:/Users/HLY/Desktop/driving_test.mp4')
        

        print('OK')
        
        
        #self.pcsv_path = r'D:\VTGNet-master1\test_data\vtgnet\data_reference.csv'
        #self.pdata = pd.read_csv(self.pcsv_path, header=None)
        #self.idx=1
        #self.interval_before = 11  # 1.5 s
        #self.interval_after = 22  # 3 s
        #self.info_st_index = 1 + 12
        #self.info_st_index_2 = (self.info_st_index + 4*(self.interval_before+1))
        ##self.info_future = self.pdata.iloc[self.idx,self.info_st_index_2:].to_numpy().reshape(-1, 4)
        #self.local_x_future = self.info_future[:, 0]
        #self.local_x_future[:]=0
        #self.local_y_future = self.info_future[:, 1]/2
        #self.st=cv2.imread((r"C:\Users\admin\Downloads\2022-04-12055615.jpg"), cv2.IMREAD_UNCHANGED)
        #self.angle=0
        self.thread = Thread(target=self.update, args=())######
        self.thread.daemon = True
        self.thread.start()
        self.thread1 = Thread(target=self.recvvv, args=())######
        self.thread1.daemon = True
        self.thread1.start()
        self.thread2 = Thread(target=self.recvvv1, args=())######
        self.thread2.daemon = True
        self.thread2.start()
        self.thread3 = Thread(target=self.recvvv2, args=())######
        self.thread3.daemon = True
        self.thread3.start()
        self.thread4 = Thread(target=self.recvvv3, args=())######
        self.thread4.daemon = True
        self.thread4.start()
        '''
        self.thread2 = Thread(target=self.recvv1, args=())######
        self.thread2.daemon = True
        self.thread2.start()
        self.thread3 = Thread(target=self.recvv2, args=())######
        self.thread3.daemon = True
        self.thread3.start()
        
        self.thread4 = Thread(target=self.recvall, args=())######
        self.thread4.daemon = True
        self.thread4.start()
        self.thread5 = Thread(target=self.recvall1, args=())######
        self.thread5.daemon = True
        self.thread5.start()
        '''
        #self.ret, self.frame = self.capture.read()
    def rotate_image(self):#good
        self.image_center = tuple(np.array(self.st.shape[1::-1]) / 2)
        self.rot_mat = cv2.getRotationMatrix2D(self.image_center, angle, 1.0)
        self.result = cv2.warpAffine(self.st, self.rot_mat, self.st.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
        
    def recvvv2(self):
        i=0
        
        while True:
            self.buf2 = b''
            self.count2=16
            while self.count2:
                self.newbuf2 =self.conn2.recv(self.count2)
                if not self.newbuf2: return None
                self.buf2 += self.newbuf2
                self.count2 -= len(self.newbuf2)
                #print('A',self.count ,len(self.newbuf))
            self.length2=self.buf2
            self.buf2 = b''
            self.count2=int(self.length2)
            #print('B',self.count)
            while int(self.count2):
                #print('B',self.length)
                self.newbuf2=self.conn2.recv(self.count2)
                if not self.newbuf2: return None
                self.buf2 += self.newbuf2
                self.count2 -= len(self.newbuf2)
                #print('B',int(self.length))
            self.stringData2=self.buf2
    def recvvv1(self):
        i=0
        
        while True:
            self.buf1 = b''
            self.count1=16
            while self.count1:
                self.newbuf1 =self.conn1.recv(self.count1)
                if not self.newbuf1: return None
                self.buf1 += self.newbuf1
                self.count1 -= len(self.newbuf1)
                #print('A',self.count ,len(self.newbuf))
            self.length1=self.buf1
            self.buf1 = b''
            self.count1=int(self.length1)
            #print('B',self.count)
            while int(self.count1):
                #print('B',self.length)
                self.newbuf1=self.conn1.recv(self.count1)
                if not self.newbuf1: return None
                self.buf1 += self.newbuf1
                self.count1 -= len(self.newbuf1)
                #print('B',int(self.length))
            self.stringData1=self.buf1
    def recvvv3(self):
        i=0
        
        while True:
            self.buf3 = b''
            self.count3=16
            while self.count3:
                self.newbuf3 =self.conn3.recv(self.count3)
                if not self.newbuf3: return None
                self.buf3 += self.newbuf3
                self.count3 -= len(self.newbuf3)
                #print('A',self.count ,len(self.newbuf))
            self.length3=self.buf3
            self.buf3 = b''
            self.count3=int(self.length3)
            #print('B',self.count)
            while int(self.count3):
                #print('B',self.length)
                self.newbuf3=self.conn3.recv(self.count3)
                if not self.newbuf3: return None
                self.buf3 += self.newbuf3
                self.count3 -= len(self.newbuf3)
                #print('B',int(self.length))
            self.stringData3=self.buf3
        
        
        
    def recvvv(self):
        i=0
        
        while True:
            self.buf = b''
            self.count=16
            while self.count:
                self.newbuf =self.conn.recv(self.count)
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
                self.newbuf=self.conn.recv(self.count)
                if not self.newbuf: return None
                self.buf += self.newbuf
                self.count -= len(self.newbuf)
                #print('B',int(self.length))
            self.stringData=self.buf
         
            #time.sleep(1)
            #print('receive:', self.data_server.decode())
            
            #print('receive1:', self.data_server1.decode())
    
    def update(self):
        i=0
        while True:
            if self.capture.isOpened():
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                self.ret, self.frame = self.capture.read()
                self.result, self.imgencode = cv2.imencode('.jpg', self.frame, encode_param)
                self.data = numpy.array(self.imgencode)
                self.stringDataS = self.data.tostring()
                self.conn.send( str(len(self.stringDataS)).ljust(16).encode());
                self.conn.send( self.stringDataS );
                print('send',i)
                i=i+1
                self.conn1.send( str(len(self.stringDataS)).ljust(16).encode());
                self.conn1.send( self.stringDataS );
                self.conn2.send( str(len(self.stringDataS)).ljust(16).encode());
                self.conn2.send( self.stringDataS );
                self.conn3.send( str(len(self.stringDataS)).ljust(16).encode());
                self.conn3.send( self.stringDataS );
                
            time.sleep(.01)

            
            
    def show_frame(self):

        if self.capture.isOpened():
            #print('receive1:', self.data_server1,'receive:', self.data_server)
            
            #cv2.imshow('frame', self.frame)
            #cv2.waitKey(1)
            #time.sleep(.1)
            #self.sp=self.sp+1
            #length = self.recvall(16)
            #print(len(length))
            #stringData = self.recvall(int(length))
            '''
            if self.sp<3:
                time.sleep(1)
            else:
            '''
            if self.stringData1 !='':
                data = numpy.fromstring(self.stringData1, dtype='uint8')
                
                decimg1=cv2.imdecode(data,1)
                '''
                cv2.imshow('CLIENT1',decimg1)
                cv2.waitKey(1)
                '''
            if self.stringData2 !='':
                data = numpy.fromstring(self.stringData2, dtype='uint8')
                decimg2=cv2.imdecode(data,1)
                '''
                cv2.imshow('CLIENT2',decimg2)
                cv2.waitKey(1)
                '''
            if self.stringData3 !='':
                data = numpy.fromstring(self.stringData3, dtype='uint8')
                decimg3=cv2.imdecode(data,1)
                '''
                cv2.imshow('CLIENT3',decimg3)
                cv2.waitKey(1)
                '''
            if (self.stringData !='') & (self.stringData1 !='') & (self.stringData2 !='') & (self.stringData3 !=''):
                
                
                data = numpy.fromstring(self.stringData, dtype='uint8')
                decimg=cv2.imdecode(data,1)
                cv2.imshow('CLIENT4',decimg)
                cv2.waitKey(1)
                result12 = cv2.addWeighted(decimg[:480][:640], 0.8, decimg1[:480][:640], 0.6, 0)
                result123 = cv2.addWeighted(result12[:480][:640], 0.8, decimg2[:480][:640], 0.6, 0)
                result1234 = cv2.addWeighted(result123[:480][:640], 0.8, decimg3[:480][:640], 0.6, 0)
                aaaa=decimg.shape
                bbbb=decimg1.shape
                cccc=decimg2.shape
                dddd=decimg3.shape
                result123= cv2.copyMakeBorder(result123, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                if aaaa[0]==640:
                    result1234[480:640][:]=decimg[480:640][:]
                if bbbb[0]==640:
                    result1234[480:640][:]=decimg1[480:640][:]
                if cccc[0]==640:
                    result1234[480:640][:]=decimg2[480:640][:]
                if dddd[0]==640:
                    result1234[480:640][:]=decimg3[480:640][:]    
                #result123= cv2.copyMakeBorder(result123, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                #result123[480:640,240:400]=self.st
                #cv2.polylines(result123, [vertices_gt], False, (0, 255, 0), 3)
                
                
                cv2.imshow('CLIENT0',result1234)
                cv2.waitKey(1)
                
                cv2.imshow('CLIENT1',decimg1)
                cv2.waitKey(1)

                cv2.imshow('CLIENT2',decimg2)
                cv2.waitKey(1)

                cv2.imshow('CLIENT3',decimg3)
                cv2.waitKey(1)

                fps = self.capture.get(cv2.CAP_PROP_FPS)
                size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')#参數是MPEG-4编码類型，文件名后缀为.avi(MPEG-4是一套用於音訊、視訊資訊的壓縮編碼標準)

                videoWriter = cv2.imWriter('21_2',result1234)#要將影片(攝影機影像還是影片)要存成圖片的話要使用cv2.imwrite()，如果想要存成影片檔的話，我們可以使用VideoWrite
                
                videoWriter.write(result1234)
                
                
                
                
                
                

            #if self.stringData2 !='' and self.stringData1 !='' and self.stringData !='':
                
        

video_stream_widget = VideoStreamWidget()
while 1:
    video_stream_widget.show_frame()
conn.close()
cv2.destroyAllWindows()