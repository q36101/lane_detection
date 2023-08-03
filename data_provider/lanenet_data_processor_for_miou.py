# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:46:30 2020

@author: mediacore
"""
"""
實現LaneNet的數據解析類
"""
import os.path as ops

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
		#self._random_dataset()
		self._next_batch_loop_count = 0

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

				gt_img_list.append(info_tmp[0])                #gt_img_list的長度會是4500，且都是gt_img的路徑
				gt_label_binary_list.append(info_tmp[1])
				gt_label_instance_list.append(info_tmp[2])

		return gt_img_list, gt_label_binary_list, gt_label_instance_list

# 	def _random_dataset(self):  #沒用到
# 		"""

# 		:return:
# 		"""
# 		assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

# 		random_idx = np.random.permutation(len(self._gt_img_list))             #打亂原本按順序0到4499的排列
# 		new_gt_img_list = []
# 		new_gt_label_binary_list = []
# 		new_gt_label_instance_list = []

# 		for index in random_idx:
# 			new_gt_img_list.append(self._gt_img_list[index])             #將打亂後的排列當作新的list，再繼續做train
# 			new_gt_label_binary_list.append(self._gt_label_binary_list[index])
# 			new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        

# 		self._gt_img_list = new_gt_img_list
# 		self._gt_label_binary_list = new_gt_label_binary_list
# 		self._gt_label_instance_list = new_gt_label_instance_list

	def next_batch(self, batch_size):
		"""

		:param batch_size:
		:return:
		"""
		assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) == len(self._gt_img_list)

		idx_start = batch_size * self._next_batch_loop_count                  
		idx_end = batch_size * self._next_batch_loop_count + batch_size                

		if idx_end > len(self._gt_label_binary_list):
			#self._random_dataset()
			self._next_batch_loop_count = 0
			return self.next_batch(batch_size)
		else:                                                    
			gt_img_list = self._gt_img_list[idx_start:idx_end]      #一開始會讀gt_img_list裡的第0跟第1項，也就是第0張圖跟第1張圖的路徑，接著2和3，4和5...
			gt_label_binary_list = self._gt_label_binary_list[idx_start:idx_end]
			gt_label_instance_list = self._gt_label_instance_list[idx_start:idx_end]

			gt_imgs = []
			gt_labels_binary = []
			gt_labels_instance = []

			for gt_img_path in gt_img_list:
				gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR))

			for gt_label_path in gt_label_binary_list:
				print(gt_label_path)
				# D:/Users/mediacore/lane_detection/data/training_data_example/tusimple_image_to_calculate_iou/label/
				print(ops.isfile(gt_label_path))
				label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
				label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
				idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
				label_binary[idx] = 1
				gt_labels_binary.append(label_binary)           #讀取第1張圖和第2張圖的值，會只有0跟1(BINARY)

			for gt_label_path in gt_label_instance_list:
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
