import numpy as np
import cv2
import os
from PIL import Image

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  
#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1 ,colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    print(type(count),count.shape)
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    precision_meam = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return precision ,precision_meam  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

#################################################################
#  标签图像文件夹
# LabelPath = 'C:/Users/HLY/Desktop/test_made/test_binary/'
# #  预测图像文件夹
# PredictPath = 'C:/Users/HLY/Desktop/model/test/sald(cfb)/'

LabelPath = 'D:/Users/mediacore/lane_detection/data/training_data_example/testing/gt_binary_image/'



#  预测图像文件夹

path = 'C:/Users/HLY/Desktop/pre(7-31)/slr_conv_3/test/'


print(path)
PredictPath = path

# PredictPath = 'D:/Users/mediacore/lane_detection/model/culane_lanenet/SALD(psan+cfb)/test/'
# PredictPath = 'D:/Users/mediacore/lane_detection/model/culane_lanenet/SALD(improve-psan=5)/test/'

#  类别数目(包括背景)
classNum = 2#16

folder1_path = LabelPath 
folder2_path = PredictPath

# 定義批次大小
batch_size = 

# 計算資料夾中的圖片數量
folder1_num_images = len(os.listdir(folder1_path))
folder2_num_images = len(os.listdir(folder2_path))


# 計算總共需要處理的批次數量
num_batches = int(folder1_num_images // batch_size)

# 初始化混淆矩陣
confusion_mat = [[0, 0], [0, 0]]
a=0
b=0
print('b',b)
# 使用迴圈進行批次處理
for batch in range(1,num_batches):
    colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

    #  获取文件夹内所有图像
    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    #  读取第一个图像，后面要用到它的shape
    Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)
    # Label0 = cv2.resize(Label0, (1280, 720),interpolation=cv2.INTER_LINEAR)
    
    #  图像数目
    label_num = len(labelList)*batch // batch_size
    
    print('label_num=',label_num)
    #  把所有图像放在一个数组里
    label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    for i in range(a,label_num):
        
        # if i >= 642 :
        #     break
        # print('i=',i)
        Label = cv2.imread(LabelPath + "//" + labelList[i])
        # Label = cv2.resize(Label, (1280, 720),interpolation=cv2.INTER_LINEAR)
        # Label = cv2.imread(LabelPath + labelList[i])
        # x,y,c=Label.shape
        # for m in range(x):
        #     for j in range(y):
        #         if Label[m,j].all()>0:
        #             Label[m,j]=255
        #         else:
        #             Label[m,j]=0
        # cv2.imshow('m',Label)
        # cv2.waitKey(0)
        Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
        label_all[i] = Label
        Predict = cv2.imread(PredictPath + "//" + PredictList[i])
        # Predict = cv2.resize(Predict, (1280, 720),interpolation=cv2.INTER_LINEAR)
        Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
        predict_all[i] = Predict
        print(LabelPath +  labelList[i])
        print(PredictPath + PredictList[i])

    #  把颜色映射为0,1,2,3...
    a = label_num
    print('a=' ,a)
    for i in range(colorDict_GRAY.shape[0]):
        label_all[label_all == colorDict_GRAY[i][0]] = i
        predict_all[predict_all == colorDict_GRAY[i][0]] = i

    #  拉直成一维
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    #  计算混淆矩阵及各精度参数
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    predict_all=0 
    label_all=0
    precision,precision_mean = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)

    # for i in range(colorDict_BGR.shape[0]):
    #     #  输出类别颜色,需要安装webcolors,直接pip install webcolors
    #     try:
    #         import webcolors
    #         rgb = colorDict_BGR[i]
    #         rgb[0], rgb[2] = rgb[2], rgb[0]
    #         print(webcolors.rgb_to_name(rgb), end = "  ")
    #     #  不安装的话,输出灰度值
    #     except:
    #         print(colorDict_GRAY[i][0], end = "  ")
    cm = confusionMatrix.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    
  




# cm = confusionMatrix.astype(np.float16)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)

    # print("")
    # print("混淆矩阵:")
    # print(confusionMatrix)
    # print("精确度:")
    # print(precision)
    # print("平均精确度:")
    # precision_mean = np.mean(precision)
    # print(precision_mean)
    # print("召回率:")
    # print(recall)
    # print("F1-Score:")
    # print(f1ccore)
    # print("整体精度:")
    # print(OA)
    # print("IoU:")
    # print(IoU)
    # print("mIoU:")
    # print(mIOU)
    # print("FWIoU:")
    # print(FWIOU)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)

    iou=TP/(TP+FP+FN)

    # print('TP:',TP)
    # print('TN:',TN)
    # print('FP:',FP)
    # print('FN:',FN)

    Precision1=TP/(TP+FP)
    recall = TP / (TP+FN)

    # print("ACC: ", ACC)

    # print("ACC_macro: ",ACC_macro)

    # print("F1-: ",F1)

    print("F1_macro:", F1_macro)
    # f1=round(F1_macro,4)
    # print(f1)
    # if f1 >0.1:
    #     f1=int(f1*100000)
    # else :
    #     f1=f1
    # print(f1)
    print('num_batches',num_batches)
    b = b+ F1_macro
    # print(type(f1))
    # print(type(b))

    # print("iou: " ,iou)

    # print("Precision1: ",Precision1)

    # print("recall: ", recall)
    print('maxf1=',b)
    print('batch=',batch)
    print('maxf1=',b/batch)

# import os
# import cv2
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# # 設定兩個資料夾的路徑
# # LabelPath = "/path/to/folder1/"
# # PredictPath = "/path/to/folder2/"

# # 讀取資料夾中的圖片並標籤資料
# folder1_images = []
# folder1_labels = []
# folder2_images = []
# folder2_labels = []

# for filename in os.listdir(LabelPath):
#     img_path = os.path.join(LabelPath, filename)
#     img = cv2.imread(img_path)
#     # 設定圖片的標籤，例如 0 表示狗
#     label = 1
#     folder1_images.append(img)
#     folder1_labels.append(label)

# for filename in os.listdir(PredictPath):
#     img_path = os.path.join(PredictPath, filename)
#     img = cv2.imread(img_path)
#     # 設定圖片的標籤，例如 1 表示貓
#     label = 1
#     folder2_images.append(img)
#     folder2_labels.append(label)

# # 將圖片轉換為模型所需的輸入格式

# # 使用您的機器學習或深度學習模型對圖片進行預測並取得預測結果

# # 計算混淆矩陣
# y_true = folder1_labels 
# y_pred = folder2_labels
# confusion_mat = confusion_matrix(y_true, y_pred)

# # 計算 Precision、Recall 和 F1 分數
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))