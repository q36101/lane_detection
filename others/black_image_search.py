# import os
# import cv2

# # Directory containing the images
# image_dir = 'C:/Users/HLY/Downloads/culane/test_binary/'

# # List to store the names of all-black images
# def find_all_black_images(directory):
#     black_images = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             file_path = os.path.join(directory, filename)
#             image = cv2.imread(file_path)
#             if image is not None:
#                 grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 if cv2.countNonZero(grayscale_image) == 0:
#                     black_images.append(file_path)
#     return black_images

# def save_paths_to_txt(file_paths, txt_file):
#     with open(txt_file, 'w') as file:
#         for path in file_paths:
#             file.write(path + '\n')

# black_images = find_all_black_images(image_dir)

# save_paths_to_txt(black_images, 'black_images.txt')

import os
import shutil

def create_new_directory(source_directory, target_directory, black_images_file):
    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # 读取包含全黑图像路径的文件
    with open(black_images_file, 'r') as file:
        black_image_paths = file.read().splitlines()
    
    # 遍历源目录中的图像文件
    for filename in os.listdir(source_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(source_directory, filename)
            
            # 如果图像路径不在全黑图像列表中，则将其复制到目标目录
            if file_path not in black_image_paths:
                shutil.copy2(file_path, target_directory)

# 指定源目录和目标目录
source_directory = 'C:/Users/HLY/Downloads/culane/test/'
target_directory = 'C:/Users/HLY/Downloads/culane/test_test/'
black_images_file = 'black_gt_images.txt'

# 创建新的目录，不包含全黑图像
create_new_directory(source_directory, target_directory, black_images_file)
