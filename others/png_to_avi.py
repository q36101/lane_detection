import cv2 as cv
import os

i=1
def image_to_video():
    file = 'C:/Users/HLY/lane_detection/data/BOSCH/color_images/color_images/train/images-2014-12-18-14-28-45_train/images-2014-12-18-14-28-45/'  # 图片目录
    output = 'data/llamas_500_train/test.mp4'  # 生成视频路径
    num = os.listdir(file)  # 生成图片目录下以图片名字为内容的列表
    height = 720
    weight = 1280#記得改
    fps = 10
    #fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') #用于avi格式的生成
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    for i in range(len(num)):
        path = file + str(i) + '.png'
        frame = cv.imread(path)
        videowriter.write(frame)

    videowriter.release()

image_to_video()