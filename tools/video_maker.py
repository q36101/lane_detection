import os
import fnmatch
import numpy as np
import cv2

#transform video to frames in tools/Folder/image
video_to_frames = "python extract_frames.py"
os.system(video_to_frames)

os.system(command_input)
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

frames_to_video = "python frames_to_video.py"
os.system(frames_to_video)



