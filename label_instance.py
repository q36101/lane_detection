import os
import fnmatch
import numpy as np
import cv2

path = "D:/YaoTeng/lanenet-lane-detection/data_for_test/temp_for_test"


height = 590
width = 1640

count = 0
for root, dirnames, filenames in os.walk(path):
	for filename in fnmatch.filter(filenames, '*.png'):
		img = cv2.imread(os.path.join(root, filename), 0)

		label_list = [0, 20, 70, 120, 170, 220]
		output = np.zeros(shape=img.shape, dtype=np.uint8)
		
		for i in range(1, len(label_list)):
			output[img == i] = label_list[i]
		
		cv2.imwrite(os.path.join(root, filename), output)
		count+=1
		
	print('Processed image number: ' + str(count))


#os.walk:列出某個資料夾底下所有的目錄和檔名
#其中第一個 root 是這行啟始路徑，
#第二個 dirNames 是一個 list，裡面包含了 root 下所有的資料夾名稱，
#而第三個 fileNames 也是一個 list，包含了 root下所有的檔案名稱。
#所以我們利用 for 走過 os.walk() 所有的 dirPath，
#並且利用底下的 for 把 fileNames 中所有的 filename拿出來，
#再以 os.path.join(root, filename) 把資料夾路徑和檔名串接起來


