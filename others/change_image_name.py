# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:52:10 2019

@author: mediacore
"""

import os
path='D:/CULane/list/training_data_example/training_data/image/' #這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
files=os.listdir(path)
file_list = []
print(files) #印出讀取到的檔名稱，用來確認自己是不是真的有讀到
print(len(files))

#%%
for i in files:
    file_list.append(int(i[0:-4]))

print(file_list)
file_list.sort(reverse=False)
print(file_list, '!!!!!!!!')


#%%
n=0 #設定初始值
for i in range(78421): #因為資料夾裡面的檔案都要重新更換名稱
	oldname=path+str(file_list[n])+'.png' #指出檔案現在的路徑名稱
	newname=path+str(n)+'.png' #在本案例中的命名規則為：年份+ - + 次序，最後一個.wav表示該檔案的型別
	os.rename(oldname,newname)
	print(oldname+'>>>'+newname) #印出原名與更名後的新名，可以進一步的確認每個檔案的新舊對應
	n=n+1 #當有不止一個檔案的時候，依次對每一個檔案進行上面的流程，直到更換完畢就會結束

#%%
import cv2
import os

image_path = 'C:/Users/mediacore/lane_detection/data/training_data_example/training_data/image_gt_binary/'
files=os.listdir(image_path)
n=0
for i in files:
    path = image_path+files[n]
    print(path)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)   #注意轉binary 和 instance時要有UNCHANGED
    flip_image = cv2.flip(image, 1)
    cv2.imwrite('C:/Users/mediacore/lane_detection/data/training_data_example/training_data/image_gt_binary_flip/' + str(4500+n) + '.png', flip_image)
    n=n+1
print('OK')

#            if random.random() < 0.25:
#                instance_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in instance_gt_labels] 
#                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]                                                  
#                image = np.concatenate((gt_imgs, binary_gt_labels, instance_gt_labels), axis=-1)     #有1/4的機率會做水平翻轉來增加data數量
#                flip_image = [cv2.flip(tmp, 1) for tmp in image]                              
#                resized_image = [cv2.resize(tmp,
#                                            dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
#                                            dst=tmp,
#                                            interpolation = cv2.INTER_NEAREST)
#                                 for tmp in flip_image]
#                resized_image = np.array(resized_image)
#                gt_imgs = resized_image[:, :, :, 0:3]
#                gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
#                binary_gt_labels = resized_image[:, :, :, 3:4]
#                instance_gt_labels = resized_image[:, :, :, 4]
