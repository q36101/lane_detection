import cv2
import os
from tqdm import tqdm
import glob
#TODO
image_folder = 'D:\video-frame*.jpg'
video_name = 'D:try-video.avi'#save as .avi
#is changeable but maintain same h&w over all  frames
width=1280
height=720 
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')#FourCC 是 4-byte 大小的碼，用來指定影像編碼方式。常見的編碼格式使用MJPG
video=cv2.VideoWriter(video_name,fourcc, 30.0, (width,height))

print(sorted(glob.glob(image_folder)))

for i in tqdm((sorted(glob.glob(image_folder)))):
     x=cv2.imread(i)
     video.write(x)

cv2.destroyAllWindows()
video.release()


#https://shengyu7697.github.io/python-opencv-save-video/
#sorted(d.items(), key=lambda x: x[1]) 中 d.items() 为待排序的对象；
#key=lambda x: x[1] 为对前面的对象中的第二维数据（即value）的值进行排序。
#key=lambda  变量：变量[维数] 。维数可以按照自己的需要进行设置。
