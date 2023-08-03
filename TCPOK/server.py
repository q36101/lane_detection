# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:31:51 2022

@author: Lee
"""

import socket
import cv2
import numpy


TCP_IP = ""
TCP_PORT = 5555
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(5)

while True:
    capture = cv2.VideoCapture('D:/Users/mediacore/lane_detection/data/test_video/2_Trim.mp4')

    conn, addr = s.accept()
    print("進入等待時間....")
    ret, frame = capture.read()
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    
    while ret:
            # 接收資料
        data_server = conn.recv(1024)
        if not data_server:  # 這裡判斷客戶端斷開的情況，不控制會無限迴圈
          print('client is lost...')
          break
        print('receive:', data_server.decode())
        
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()
    
        conn.send( str(len(stringData)).ljust(16).encode());
        conn.send( stringData );
    
        ret, frame = capture.read()
        decimg=cv2.imdecode(data,1)
        cv2.imshow('SERVER2',decimg)
        cv2.waitKey(30)
        
conn.close()
cv2.destroyAllWindows()

