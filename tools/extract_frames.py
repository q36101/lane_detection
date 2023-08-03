# Program To Read video 
# and Extract Frames 
import numpy as np
import cv2 
import os

image_path = "D:/Users/mediacore/lane_detection/data/frames_lane_jpg/"
  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
   
    os.chdir(image_path)
  
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        try:
        
        # Saves the frames with frame-count 
            cv2.imwrite('im0.png',image) 
        except cv2.error:
            print('error')
        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("D:/Users/mediacore/lane_detection/data/tusimple_test_image/lane.mp4") 

#https://www.jianshu.com/p/949683764115
#os.chdir() 方法用于改变当前工作目录到指定的路径

#查看当前工作目录
#retval = os.getcwd()
#print "当前工作目录为 %s" % retval