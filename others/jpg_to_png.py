import os
import fnmatch
import numpy as np
import cv2
from PIL import Image

path = "C:/Users/mediacore/lane_detection/data/frames_lane_jpg"
height = 720
width = 1280

counter = 0

for root, dirnames, filenames in os.walk(path):
	for filename in fnmatch.filter(filenames, '*.jpg'):
		img = Image.open(os.path.join(root, filename))

		
		png_filename = os.path.join(filename).split('jpg',1)[0] + 'png'
		img.save(os.path.join('C:/Users/mediacore/lane_detection/data/frames_lane', png_filename))
		
		counter += 1
		print(counter)	
		
print(counter)
#home/ycw/CULane/training_data/image/driver_23_30frame/05151640_0419.MP4/00000.jpg




