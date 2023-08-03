# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:31:51 2022

@author: Lee
"""

import socket
import cv2
import numpy

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

#TCP_IP = "169.254.14.65"
#TCP_IP="192.168.50.113"
#TCP_IP="192.168.43.211"
TCP_IP='localhost'
TCP_PORT = 5555

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

while 1:


    length = recvall(sock,16)
    stringData = recvall(sock, int(length))
    #sock.send(stringData)
    sock.send( str(len(stringData)).ljust(16).encode());
    sock.send( stringData );
    data = numpy.fromstring(stringData, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    #cv2.imshow('CLIENT1',decimg)
    #cv2.waitKey(1)

sock.close()
cv2.destroyAllWindows()
