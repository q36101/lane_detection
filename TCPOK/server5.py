# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:31:51 2022

@author: Lee
"""
from threading import Thread
import socket
import cv2, time
import numpy






class VideoStreamWidget(object):
    def __init__(self, src=0):
        
        self.TCP_IP = ""
        self.TCP_PORT = 5555
        self.sp=0
        self.stringData=''
        self.stringData1=''
        self.stringData2=''
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.TCP_IP, self.TCP_PORT))
        self.s.listen(True)
        self.conn, self.addr = self.s.accept()
        self.conn1, self.addr1 = self.s.accept()
        self.conn2, self.addr2 = self.s.accept()
        self.capture = cv2.VideoCapture(src)
        print('OK')
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
    '''
    def recvv1(self):
        while True:
            self.data_server = self.conn.recv(1024)
            #print('receive:', self.data_server.decode())
            #self.data_server1 = self.conn1.recv(1024)
            #print('receive1:', self.data_server1.decode())
    def recvv2(self):
        while True:
            #self.data_server = self.conn.recv(1024)
            #print('receive:', self.data_server.decode())
            self.data_server1 = self.conn1.recv(1024)
            #print('receive1:', self.data_server1.decode())


    def recvall(self, count):
        self.buf = b''
        while count:
            self.newbuf = self.conn.recv(count)
            if not self.newbuf: return None
            self.buf += self.newbuf
            self.count -= len(self.newbuf)
    
    def recvall1(self, count):
        self.buf1 = b''
        while count:
            self.newbuf1 = self.conn1.recv(count)
            if not self.newbuf1: return None
            self.buf1 += self.newbuf1
            self.count1 -= len(self.newbuf1)
    '''
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
            if self.stringData !='':
                data = numpy.fromstring(self.stringData, dtype='uint8')
                decimg=cv2.imdecode(data,1)
                #print(data.shape)
                #print("-----------")
                #print(decimg)
                cv2.imshow('CLIENT2',decimg)
                cv2.waitKey(1)
            if self.stringData1 !='':
                data = numpy.fromstring(self.stringData1, dtype='uint8')
                decimg1=cv2.imdecode(data,1)
                #print(data.shape)
                #print("-----------")
                #print(decimg)
                cv2.imshow('CLIENT21',decimg1)
                cv2.waitKey(1)
            if self.stringData2 !='':
                data = numpy.fromstring(self.stringData2, dtype='uint8')
                decimg2=cv2.imdecode(data,1)
                #print(data.shape)
                #print("-----------")
                #print(decimg)
                cv2.imshow('CLIENT22',decimg2)
                cv2.waitKey(1)
        

video_stream_widget = VideoStreamWidget()
while 1:
    video_stream_widget.show_frame()
conn.close()
cv2.destroyAllWindows()