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
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.TCP_IP, self.TCP_PORT))
        self.s.listen(True)
        self.conn, self.addr = self.s.accept()
        self.conn1, self.addr1 = self.s.accept()
        self.capture = cv2.VideoCapture('C:/Users/Mini/Desktop/lane_det/out.mp4')
        self.thread = Thread(target=self.update, args=())######
        self.thread.daemon = True
        self.thread.start()
        self.thread1 = Thread(target=self.recvvv, args=())######
        self.thread1.daemon = True
        self.thread1.start()
        self.thread2 = Thread(target=self.recvv1, args=())######
        self.thread2.daemon = True
        self.thread2.start()
        self.thread3 = Thread(target=self.recvv2, args=())######
        self.thread3.daemon = True
        self.thread3.start()
        '''
        self.thread4 = Thread(target=self.recvall, args=())######
        self.thread4.daemon = True
        self.thread4.start()
        self.thread5 = Thread(target=self.recvall1, args=())######
        self.thread5.daemon = True
        self.thread5.start()
        '''
        self.ret, self.frame = self.capture.read()
        
        
        
    def recvvv(self):
        while True:
            self.buf = b''
            self.count=16
            while self.count:
                self.newbuf =self.conn.recv(self.count)
                if not self.newbuf: return None
                self.buf += self.newbuf
                self.count -= len(self.newbuf)
            self.length=self.buf
            while int(self.length):
                self.newbuf=self.conn.recv(count)
                if not self.newbuf: return None
                self.buf += self.newbuf
                self.count -= len(self.newbuf)
            self.stringData=self.buf
            #print('receive:', self.data_server.decode())
            
            #print('receive1:', self.data_server1.decode())
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
    '''
    def recvall1(self, count):
        self.buf1 = b''
        while count:
            self.newbuf1 = self.conn1.recv(count)
            if not self.newbuf1: return None
            self.buf1 += self.newbuf1
            self.count1 -= len(self.newbuf1)
    '''
    def update(self):
        while True:
            if self.capture.isOpened():
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                self.ret, self.frame = self.capture.read()
                self.result, self.imgencode = cv2.imencode('.jpg', self.frame, encode_param)
                self.data = numpy.array(self.imgencode)
                self.stringData = self.data.tostring()
                self.conn.send( str(len(self.stringData)).ljust(16).encode());
                self.conn.send( self.stringData );
                self.conn1.send( str(len(self.stringData)).ljust(16).encode());
                self.conn1.send( self.stringData );
                
            time.sleep(.01)

            
            
    def show_frame(self):

        if self.capture.isOpened():
            #print('receive1:', self.data_server1,'receive:', self.data_server)
            
            cv2.imshow('frame', self.frame)
            cv2.waitKey(1)
            
            
            #length = self.recvall(16)
            #print(len(length))
            #stringData = self.recvall(int(length))
            data = numpy.fromstring(self.stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
            #print(data.shape)
            print("-----------")
            #print(decimg)
            cv2.imshow('CLIENT2',decimg)
            cv2.waitKey(1)
        

video_stream_widget = VideoStreamWidget()
while 1:
    video_stream_widget.show_frame()
conn.close()
cv2.destroyAllWindows()


def trans():
    TCP_IP="192.168.19.1"
    TCP_PORT = 5555

    sock = socket.socket()
    sock.connect((TCP_IP, TCP_PORT))
    while 1:
        length = vd.recvall(sock,16)
        stringData = vd.recvall(sock, int(length))
        data = numpy.fromstring(stringData, dtype='uint8')
        decimg=cv2.imdecode(data,1)
    return decimg


def main():
    TCP_IP="192.168.19.1"
    TCP_PORT = 5555

    sock = socket.socket()
    sock.connect((TCP_IP, TCP_PORT))
    args = vd.init_args()
    thread1=vd.test_lanenet(video_path=args.video_path, weights_path=args.weights_path, use_gpu=args.use_gpu, net_flag=args.net_flag)
    thread2=Thread(vd.trans)
    thread2.start()
    while 1:       
        try:
            stringData =thread1()
            sock.send(str(len(stringData)).ljust(16).encode())
            sock.send(stringData)
        except AttributeError:
            pass


if __name__ == '__main__':
    main()