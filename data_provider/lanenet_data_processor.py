 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time	   : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site	   : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File	   : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
實現LaneNet的數據解析類
"""
import os.path as ops
import tensorflow as tf
import cv2
import numpy as np
import itertools 
from config import global_config

try:
	from cv2 import cv2
except ImportError:
	pass

CFG = global_config.cfg

class DataSet(object):
	"""
	實現數據集類
	"""

	def __init__(self, dataset_info_file):
		"""

		:param dataset_info_file:
		"""
		self._gt_img_list, self._gt_label_binary_list, \
		self._gt_label_instance_list = self._init_dataset(dataset_info_file)
		self._random_dataset()
		self._next_batch_loop_count = 0
		self._gt_img_list_new = []
		self._gt_label_binary_list_new = []
		self._gt_label_instance_list_new = []
		self._img_list = []
		self._label_binary_list = []
		self._label_instance_list = []

	def _init_dataset(self, dataset_info_file):
		"""

		:param dataset_info_file:
		:return:
		"""
		gt_img_list = []
		gt_label_binary_list = []
		gt_label_instance_list = []

		assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

		with open(dataset_info_file, 'r') as file:           #讀圖片
			for _info in file:
				info_tmp = _info.strip(' ').split()
				gt_img_list.append(info_tmp[0])                #gt_img_list的長度會是9000，且都是gt_img的路徑(train的情況)
				
				gt_label_binary_list.append(info_tmp[1])
				gt_label_instance_list.append(info_tmp[2])
                
#				print(len(gt_img_list), 'AAAAAAAAAAAAA')
# 				print(len(gt_label_binary_list), 'BBBBBBBBBB')
# 				print(len(gt_label_instance_list), 'CCCCCCCCCCCC')
                
			print(len(gt_img_list), '!!!!!!!!!!!')
			print(len(gt_label_binary_list), '@@@@@@@@@@@@@')
			print(len(gt_label_instance_list), '##############')

		return gt_img_list, gt_label_binary_list, gt_label_instance_list

	def _random_dataset(self):
		"""

		:return:
		"""
		assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

		random_idx = np.random.permutation(len(self._gt_img_list))             #打亂原本按順序0到4499的排列
		new_gt_img_list = []
		new_gt_label_binary_list = []
		new_gt_label_instance_list = []

		for index in random_idx:
			new_gt_img_list.append(self._gt_img_list[index])             #將打亂後的排列當作新的list，再繼續做train
			new_gt_label_binary_list.append(self._gt_label_binary_list[index])
			new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        

		self._gt_img_list = new_gt_img_list
		self._gt_label_binary_list = new_gt_label_binary_list
		self._gt_label_instance_list = new_gt_label_instance_list

	def next_batch(self, batch_size):
		"""

		:param batch_size:
		:return:
		"""
		assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) == len(self._gt_img_list)

		idx_start = batch_size * self._next_batch_loop_count                  
		idx_end = batch_size * self._next_batch_loop_count + batch_size                
            
		if self._next_batch_loop_count == 0:
                                    
			for i in range(len(self._gt_img_list)-3):                     #根據不同情況要改FOR迴圈跑的次數
			    A_chose_B_img_list = list(itertools.combinations(self._gt_img_list[i:i+5], CFG.TRAIN.T))
			    A_chose_B_binary_list = list(itertools.combinations(self._gt_label_binary_list[i:i+5], CFG.TRAIN.T))           #3張取2張組合
			    A_chose_B_instance_list = list(itertools.combinations(self._gt_label_instance_list[i:i+5], CFG.TRAIN.T))
			    self._gt_img_list_new.extend(A_chose_B_img_list)
			    self._gt_label_binary_list_new.extend(A_chose_B_binary_list)
			    self._gt_label_instance_list_new.extend(A_chose_B_instance_list)
            
			for j in range(len(self._gt_img_list_new)):
			    self._img_list.append(self._gt_img_list_new[j][0])
			    self._img_list.append(self._gt_img_list_new[j][1])
			    self._img_list.append(self._gt_img_list_new[j][2])
			    self._img_list.append(self._gt_img_list_new[j][3])
			    # self._img_list.append(self._gt_img_list_new[j][4])
			    # self._img_list.append(self._gt_img_list_new[j][5])
			    # self._img_list.append(self._gt_img_list_new[j][6])
			    # self._img_list.append(self._gt_img_list_new[j][7])			    
			    self._label_binary_list.append(self._gt_label_binary_list_new[j][0])
			    self._label_binary_list.append(self._gt_label_binary_list_new[j][1])
			    self._label_binary_list.append(self._gt_label_binary_list_new[j][2])
			    self._label_binary_list.append(self._gt_label_binary_list_new[j][3])
			    # self._label_binary_list.append(self._gt_label_binary_list_new[j][4])
			    # self._label_binary_list.append(self._gt_label_binary_list_new[j][5])
			    # self._label_binary_list.append(self._gt_label_binary_list_new[j][6])
			    # self._label_binary_list.append(self._gt_label_binary_list_new[j][7])
			    self._label_instance_list.append(self._gt_label_instance_list_new[j][0])
			    self._label_instance_list.append(self._gt_label_instance_list_new[j][1])
			    self._label_instance_list.append(self._gt_label_instance_list_new[j][2])
			    self._label_instance_list.append(self._gt_label_instance_list_new[j][3])
			    # self._label_instance_list.append(self._gt_label_instance_list_new[j][4])
			    # self._label_instance_list.append(self._gt_label_instance_list_new[j][5])
			    # self._label_instance_list.append(self._gt_label_instance_list_new[j][6])
			    # self._label_instance_list.append(self._gt_label_instance_list_new[j][7])

		if idx_end > len(self._img_list):
			# self._random_dataset()
			self._next_batch_loop_count = 0
			self._gt_img_list_new = []
			self._gt_label_binary_list_new = []
			self._gt_label_instance_list_new = []
			self._img_list = []
			self._label_binary_list = []
			self._label_instance_list = []            
			return self.next_batch(batch_size)
		else:        
			# self._random_dataset()                                            
			gt_img_list = self._img_list[idx_start:idx_end]      #一開始會讀gt_img_list裡的第0跟第1項，也就是第0張圖跟第1張圖的路徑，接著2和3，4和5...
			gt_label_binary_list = self._label_binary_list[idx_start:idx_end]
			gt_label_instance_list = self._label_instance_list[idx_start:idx_end]

			gt_imgs = []
			gt_labels_binary = []
			gt_labels_instance = []

			for gt_img_path in gt_img_list:
				gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR))
				# print('gt_label_path=',gt_img_path)


			for gt_label_path in gt_label_binary_list:
				print('gt_label_path=',gt_label_path)
				# print('ops.isfile(gt_label_path)=',ops.isfile(gt_label_path))
				label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
				label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
				idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
				label_binary[idx] = 1
				gt_labels_binary.append(label_binary)           #讀取第1張圖和第2張圖的值，會只有0跟1(BINARY)

			for gt_label_path in gt_label_instance_list:
				# print('gt_label_path=',gt_label_path)
				label_img = cv2.imread(gt_label_path, cv2.IMREAD_UNCHANGED)
				gt_labels_instance.append(label_img)
                

			self._next_batch_loop_count += 1
			return gt_imgs, gt_labels_binary, gt_labels_instance


if __name__ == '__main__':
	val = DataSet('/home/baidu/DataBase/Semantic_Segmentation/Kitti_Vision/data_road/lanenet_training/train.txt')
	a1, a2, a3 = val.next_batch(1)
	cv2.imwrite('test_binary_label.png', a2[0] * 255)
	b1, b2, b3 = val.next_batch(50)
	c1, c2, c3 = val.next_batch(50)
	dd, d2, d3 = val.next_batch(50)
	#如果要讓內部屬性不被外部訪問，可以把屬性的名稱前加上兩個下劃線"_gt_img_list"-->"__gt_img_list"，在Python中，例項的變數名如果以開頭，就變成了一個私有變數（private），只有內部可以訪問，外部不能訪問
	#需要注意的是，在Python中，變數名類似__xxx__的，也就是以雙下劃線開頭，並且以雙下劃線結尾的，是特殊變數，特殊變數是可以直接訪問的，不是private變數，所以，不能用__name__、__score__這樣的變數名。
	#有些時候，你會看到以一個下劃線開頭的例項變數名，比如_name，這樣的例項變數外部是可以訪問的，但是，按照約定俗成的規定，當你看到這樣的變數時，意思就是，“雖然我可以被訪問，但是，請把我視為私有變數，不要隨意訪問”。