from PIL import Image
import numpy as np
import cv2 as cv
import cv2 as cv2
import os

# i=1
# while i <=11000:

#     path = 'C:/Users/HLY/lane_detection/data/BOSCH/images-2014-12-22-15-18-11_instance/'+str(i)+'.png'
#     im = Image.open(path)     
#     img = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
#     # 获得行数和列数即图片大小
#     rowNum, colNum = img.shape[:2]

#     for x in range(0, rowNum):
#         for y in range(0, colNum):
#                 # 在opencv里b和r通道刚好是反着的， 比如通过句柄精灵获得图片中某个颜色的rgb值为（204,255,255），在使用opencv时需要将img[x, y].tolist() == [xxx, xxx, xxx]中 == 右边的值改为[255, 255, 204]，下面是把所有想改变的颜色变为了白色
#             if  img[x, y].tolist() != [0,0,0] :
#                 img[x, y] = np.array([255, 255, 255])
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         # 保存修改后图片
#     cv.imwrite('C:/Users/HLY/lane_detection/data/BOSCH/images-2014-12-22-15-18-11_binary/'+str(i)+'.png', img)
#     print(i)
#     i=i+1

# 讀取原始圖片
input_dir = 'C:/Users/HLY/Downloads/culane/val_instance/'
output_dir = 'C:/Users/HLY/Downloads/culane/val_instance1/'
for filename in os.listdir(input_dir):
    # 確定檔案是圖像檔案
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 讀取原始圖片
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # # 將非黑色部分轉換為白色
        # image[image > 0] = 255
        image = image*50


        # 儲存灰度圖像到輸出目錄中
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

        print(f'Processed: {filename}')

print('Image processing complete.')
