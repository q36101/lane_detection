import os
import fnmatch
import numpy as np
import cv2

path = "/home/ycw/anaconda2/fpn-lane-detection/tools/Folder"
file_list = list()

for root, dirnames, filenames in os.walk(path):
	for dirname in fnmatch.filter(dirnames, '*image'):
	#for dirname in fnmatch.filter(dirnames, '*.MP4'):
		file_list.append(os.path.join(dirname))
		
					
print (file_list)

command_for_testing = "python test_lanenet.py --is_batch True --batch_size 2 --save_dir data/tusimple_test_image/ret --weights_path /home/ycw/anaconda2/fpn-lane-detection/model/tusimple_lanenet/culane_lanenet_vgg_2018-11-04-00-51-03.ckpt-44000 --image_path /home/ycw/anaconda2/fpn-lane-detection/tools/Folder/"
ret_path = "/home/ycw/anaconda2/fpn-lane-detection/tools/data/tusimple_test_image/"

path_to_tools = "/home/ycw/anaconda2/fpn-lane-detection/tools/"
path_to_tusimple_test_image = "/home/ycw/anaconda2/fpn-lane-detection/tools/data/tusimple_test_image/"

for i in range(len(file_list)):
	command_input = command_for_testing + file_list[i]
	os.system(command_input)
	os.chdir(path_to_tusimple_test_image)
	os.rename("ret", file_list[i])
	os.chdir(path_to_tools)

#height = 720
#width = 1280

#count = 0
#for root, dirnames, filenames in os.walk(path_to_tusimple_test_image):
#	for filename in fnmatch.filter(filenames, '*.png'):
#		img = cv2.imread(os.path.join(root, filename), 0)
#
#		label_list = [0, 255]
#		output = np.zeros(shape=img.shape, dtype=np.uint8)
#		
#		for i in range(1, len(label_list)):
#			output[img == i] = label_list[i]
#		
#		cv2.imwrite(os.path.join(root, filename), output)
#		count+=1
#		
#	print('Processed image number: ' + str(count))
#
#
#os.system('python calculate_accuracy.py')



