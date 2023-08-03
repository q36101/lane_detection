       #1.导入库
import matplotlib.pyplot as plt       #Matplotlib 是一个 Python 的 2D绘图库
import pylab
import matplotlib.image as img
import numpy as np                     #NumPy是Python的一种开源的数值计算扩展
import cv2
import os
     #2.读取图片并输出状态：得到x和y轴的数值，对图片进行拷贝（copy）
img=img.imread('D:/Users/mediacore/lane_detection/data/tusimple_test_image/roi/0.png')
# print('This image is :',type(image),'with dimensions:',image.shape)
# plt.imshow(image)          #显示图像
# plt.show()

img=img[int(img.shape[0]*1/4) : int(img.shape[0]) , int(img.shape[1]*0) : int(img.shape[1])]#int(img.shape[1]*1/40)
# black[int(img.shape[0]*18/20) : int(img.shape[0]*19/20) , int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]=img1
img=img[int(img.shape[0]*0) : int(img.shape[0]*1/3) , int(img.shape[1]*0) : int(img.shape[1])]

cv2.imshow('frame2', img)
print(img.shape)
cv2.waitKey(0)


# ysize=image.shape[0]                   #获取x轴和y轴并对图片进行拷贝
# xsize=image.shape[1]
# color_select=np.copy(image)         #用copy拷贝而不是用‘=’
# line_image = np.copy(image)

      #3.接下来，我在变量red_threshold，green_threshold和blue_threshold中定义颜色的阈值，并用这些值填充rgb_threshold。 这个矢量包含我在选择时允许的红色，绿色和蓝色（R，G，B）的最小值。
# red_threshold=0.95
# green_threshold=0.5
# blue_threshold=0.5    #若红绿蓝的阈值都设置为0，则说明图片的全部像素都被选中
# rgb_threshold=[red_threshold,green_threshold,blue_threshold]

#    #4.接下来，我将选择阈值以下的任何像素并将其设置为零。之后，符合我的颜色标准(高于阈值)的所有像素将被保留，而那些不符合阈值的像素将被黑掉。
# #用布尔值或者'|'来识别低于阈值的像元

# thresholds= (image[:,:,0]<rgb_threshold[0]) | (image[:,:,1]<rgb_threshold[1]) | (image[:,:,2]<rgb_threshold[2])

# color_select[thresholds]=[0,0,0]       #color_select 是选择像素高于阈值的结果，低于阈值的显示黑色。

# plt.imshow(color_select)          #显示图像

# plt.show()

# region_select = np.copy(image)

# left_bottom = [0, 720]
# right_bottom = [1280, 720]
# left_top = [1, 200]
# right_top = [1279, 200]
# apex = [1, 200]

# fit_left = np.polyfit((left_bottom[0], left_top[0]), (left_bottom[1], left_top[1]), 1)
# fit_right = np.polyfit((right_bottom[0], right_top[0]), (right_bottom[1], right_top[1]), 1)
# fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
# fit_top = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)


# XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

# region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
#                     (YY > (XX*fit_right[0] + fit_right[1])) & \
#                     (YY < (XX*fit_bottom[0] + fit_bottom[1])) &\
#                     (YY < (XX*fit_top[0] + fit_top[1]))

# region_select[region_thresholds] = [255, 0, 0]



# image = region_select

# color_thresholds= (image[:,:,0]<rgb_threshold[0]) | (image[:,:,1]<rgb_threshold[1]) | (image[:,:,2]<rgb_threshold[2])

# color_select[color_thresholds]=[0,0,0]

# plt.imshow(color_select)
# plt.show()

# plt.imshow(region_select)
# plt.show()

# line_image[~color_thresholds & region_thresholds] = [255,0,0]

# plt.imshow(line_image)
# plt.show()

# left_bottom = [0, 720]
# right_bottom = [1280, 720]
# left_top = [0, 200]
# right_top = [1280, 200]
# apex = [400, 0]

# fit_left = np.polyfit((left_bottom[0], left_top[0]), (left_bottom[1], left_top[1]), 1)
# fit_right = np.polyfit((right_bottom[0], right_top[0]), (right_bottom[1], right_top[1]), 1)
# fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# fit_top = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)

# XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

# region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
#                     (YY > (XX*fit_right[0] + fit_right[1])) & \
#                     (YY < (XX*fit_bottom[0] + fit_bottom[1])) &\
#                     (YY < (XX*fit_top[0] + fit_top[1]))