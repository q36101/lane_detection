import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse


def walkThroughDataset(dataSetDir):
    trainIndexFilePath = os.path.join(dataSetDir, r'list\val_gt.txt')

    trainFilePairList = []
    n = 0

    with open(trainIndexFilePath, 'r') as f:
        for line in f.readlines():
            imagePath, segImagePath, lane0, lane1, lane2, lane4 = line.split()
            trainFilePairList.append(
                (os.path.join(dataSetDir, imagePath[1:]), os.path.join(dataSetDir, segImagePath[1:]), lane0, lane1, lane2, lane4))

    colorMapMat = np.zeros((5, 3), dtype=np.uint8)

    for i in range(0, 5):
        if i == 1:
            #colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)  會長這樣=> [0~255隨機取一, 0~255隨機取一, 0~255隨機取一]
            colorMapMat[i] = [255, 255, 255] #[20, 20, 20]
        elif i == 2:
            colorMapMat[i] = [255, 255, 255] #[70, 70, 70]
        elif i == 3:
            colorMapMat[i] = [255, 255, 255] #[120, 120, 120]
        elif i == 4:
            colorMapMat[i] = [255, 255, 255] #[170, 170, 170]
        elif i == 0:
            colorMapMat[i] = [0, 0, 0]

    for imageFile, segFile, _, _, _, _ in tqdm(trainFilePairList):
        img_bgr = cv2.imread(imageFile)     #原彩圖
        seg = cv2.imread(segFile,cv2.IMREAD_UNCHANGED)
        segImage = colorMapMat[seg]      #GT圖

        res = cv2.addWeighted(img_bgr, 0.7, segImage, 0.7, 0.4)
        cv2.imshow('CULane Dataset Quick Inspector', img_bgr)
        cv2.imwrite('D:/CULane/list/training_data_example/validation_data/image/' + str(n) + '.png', img_bgr)      #存圖片
        n = n + 1
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyWindow('CULane Dataset Quick Inspector')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CULane Dataset Quick Inspector')
    parser.add_argument('--rootDir', type=str, default=r'D:\CULane',
                        help='root directory (default: D:\\CULane)')
    args = parser.parse_args()                    
    return args

if __name__ == '__main__':
    args = parse_args()
    walkThroughDataset(args.rootDir)
