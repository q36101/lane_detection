import cv2
import os
import numpy as np

def convert_image_to_white(image_path, save_directory):
    image = cv2.imread(image_path)
    if image is not None:
        mask = np.all(image != [0, 0, 0], axis=-1)  # 創建遮罩
        result = image.copy()
        result[mask] = [255, 255, 255]  # 將遮罩內的部分轉為白色
        save_path = os.path.join(save_directory, os.path.basename(image_path))
        cv2.imwrite(save_path, result)
        print(f"Processed image: {image_path}")
    else:
        print(f"Failed to read image: {image_path}")

def convert_image_to_gray(image_path, save_directory):
    image = cv2.imread(image_path)
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 將圖片轉換為灰階
        save_path = os.path.join(save_directory, os.path.basename(image_path))
        cv2.imwrite(save_path, gray_image)
        print(f"Processed image: {image_path}")
    else:
        print(f"Failed to read image: {image_path}")

# 處理圖像的資料夾路徑
image_directory = 'C:/Users/HLY/Downloads/culane/train_instance/'

# 儲存處理後圖像的資料夾路徑
save_directory = 'C:/Users/HLY/Downloads/culane/train_instance/'

# 創建儲存資料夾
os.makedirs(save_directory, exist_ok=True)

# 讀取圖像並進行處理
for i in range(0,88880):#
    image_path = os.path.join(image_directory, f"{i+1}.png")
    # convert_image_to_white(image_path, save_directory)
    convert_image_to_gray(image_path, save_directory)