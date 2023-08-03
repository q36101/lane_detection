import cv2
import os
import numpy as np
import shutil
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
try:
    from utils.common import warnings
except ImportError:
    import warnings


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=1, thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape[:2], width=width) for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape[:2], width=width) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            inter = (x * y).sum()
            ious[i, j] = inter / (x.sum() + y.sum() - inter)
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


# def remove_consecutive_duplicates(x):
#     """Remove consecutive duplicates"""
#     y = []
#     for t in x:
#         if len(y) > 0 and y[-1] == t:
#             warnings.warn('Removed consecutive duplicate point ({}, {})!'.format(t[0], t[1]))
#             continue
#         y.append(t)
#     return y
# def remove_consecutive_duplicates(x):
#     """Remove consecutive duplicates"""
#     y = []
#     for t in x:
#         if len(y) > 0 and np.all(y[-1] == t):
#             warnings.warn('Removed consecutive duplicate point ({}, {})!'.format(t[0], t[1]))
#             continue
#         y.append(t)
#     return y
def remove_consecutive_duplicates(x):
    y = []
    for t in x:
        if len(y) > 0 and np.all(y[-1] == t):
            continue
        y.append(t)
    return y

def interp(points, n=50):
    if points is None:
        return np.array([])
    if len(points) == 2:
        return np.array(points)

    points = remove_consecutive_duplicates(points)



    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # 獲取原始點的索引
    indices = np.arange(len(points))
    
    # 創建新的點的索引，用於線性插值
    new_indices = np.linspace(0, len(points)-1, n)
    
    # 進行線性插值
    new_x = np.interp(new_indices, indices, x)
    new_y = np.interp(new_indices, indices, y)
    
    return np.column_stack((new_x, new_y))
    # tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    # u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    # return np.array(splev(u, tck)).T

def culane_metric(pred, anno, width=30, iou_threshold=0.5, official=True, img_shape=(590, 1640, 3)):
    # print(type(pred),np.shape(pred))
    # print(type(anno),np.shape(anno))
    if len(pred) == 0:
        return 0, 0, len(anno), np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    if len(anno) == 0:
        return 0, len(pred), 0, np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    # print(type(pred),np.shape(pred))
    # print(type(anno),np.shape(anno))
    # print(interp(pred))
    
    interp_pred = [interp(pred_lane) for pred_lane in pred]  # (4, 50, 2)
    interp_anno = [interp(anno_lane) for anno_lane in anno]  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    print('ious',ious)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    pred_ious = np.zeros(len(pred))
    print('pred_ious',pred_ious)
    pred_ious[row_ind] = ious[row_ind, col_ind]
    # print()
    return tp, fp, fn, pred_ious, pred_ious > iou_threshold

# 設定其他相關參數
width = 30  # 車道線寬度
iou_threshold = 0.5  # IoU 閾值
official = True  # 是否使用官方的離散交叉 IoU 評估方法
img_shape = (590, 1640, 3)  # 圖片尺寸

# 讀取圖片目錄下的所有圖片
pred_path0 = 'C:/Users/HLY/Desktop/pre(7-12)/culane/test_174000'
# pred_path1 = 'C:/Users/HLY/Desktop/pre(7-12)/culane/test_6000'
# pred_path = 'C:/Users/MediacoreSquare/Desktop/pre(6-16)/pool_sac_batch4_lr3_rearrange/test'
# gt_path = 'D:/lane_detection/data/training_data_example/tu_train/testing/gt_binary_image'
# pred_images = []
# for i in range(1, 1000):
#     img = cv2.imread(os.path.join(pred_path, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
#     pred_images.append(img)

# # 讀取真實圖片
gt_path = 'C:/Users/HLY/Downloads/culane/test_binary'
# gt_path = 'C:/Users/HLY/lane_detection/data/BOSCH/images-2014-12-22-12-35-10_binary/'
# gt_images = []
# for i in range(1, 1000):
#     img = cv2.imread(os.path.join(gt_path, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
#     gt_images.append(img)

# # 使用 culane_metric 函式進行車道標記分割性能評估
# tp, fp, fn, pred_ious, pred_iou_thresholds = culane_metric(pred_images, gt_images, width, iou_threshold, official, img_shape)

# # 輸出評估結果
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1_score = 2 * (precision * recall) / (precision + recall)culane_metrio

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1_score}")
# pred_paths = [
#     'C:/Users/HLY/Desktop/pre(7-12)/culane/test_74000',
#     'C:/Users/HLY/Desktop/pre(7-12)/culane/test_6000'
# ]
batch_size = 1  # Number of images to process in each batch
total_images = 34680  # Total number of images to process
results = []  # List to store results
low_f1_images = []
best_f1_score = 0.0
best_pred_path = ''
# bad_images_path = r'C:/Users/HLY/Desktop/pre(7-12)/culane/bad/'  # 設定儲存壞圖片的資料夾路徑
# Load and process images in batches
for batch_start in range(0, total_images, batch_size):
    batch_end = min(batch_start + batch_size, total_images)

    # Load the batch of predicted and ground truth images
    pred_images0 = []
    pred_images1 = []
    gt_images = []
    for i in range(batch_start + 1, batch_end + 1):
        pred_img0 = cv2.imread(os.path.join(pred_path0, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        pred_images0.append(pred_img0)
        # pred_img1 = cv2.imread(os.path.join(pred_path1, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        # pred_images1.append(pred_img1)
        
        gt_img = cv2.imread(os.path.join(gt_path, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        gt_images.append(gt_img)
        print(i)
    print('batch_start')

    # Calculate metrics for the batch
    tp, fp, fn, pred_ious, pred_iou_thresholds1 = culane_metric(pred_images0, gt_images, width, iou_threshold, official, img_shape)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision',precision)
    print('recall',recall)
    # tp, fp, fn, pred_ious, pred_iou_thresholds2 = culane_metric(pred_images1, gt_images, width, iou_threshold, official, img_shape)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # print(precision)
    # print(recall)
    # print('pred_iou_thresholds1',pred_iou_thresholds1)
    # print('pred_iou_thresholds2',pred_iou_thresholds2)
    # if pred_iou_thresholds1.all() > pred_iou_thresholds2.all():
    pred_iou_thresholds = pred_iou_thresholds1
    # else :
    #     pred_iou_thresholds = pred_iou_thresholds2

    if (precision + recall) == 0:
            f1_score = 0  # Assign a default value or handle it appropriately
    else:
            f1_score = 2 * (precision * recall) / (precision + recall)
    # f1_score = 2 * (precision * recall) / (precision + recall)

    # Save the results of the batch
    results.extend(list(pred_iou_thresholds1))
    # if f1_score < 0.6:
    #     for i in range(batch_start + 1, batch_end + 1):
    #         img_path = os.path.join(pred_path1, f"{i}.png")
    #         dest_path = os.path.join(bad_images_path, f"{i}.png")
    #         shutil.copy(img_path, dest_path)
            # print(f"Image {i} is saved in the bad images folder")
    # if f1_score < 0.52:
    #     low_f1_images.extend([os.path.join(pred_path, f"{i}.png") for i in range(batch_start + 1, batch_end + 1)])

    # Print the progress
    print(f"Processed images: {batch_start + 1}-{batch_end}, F1-Score: {f1_score}")

# Calculate the overall F1-Score
overall_precision = results.count(True) / total_images
overall_recall = results.count(True) / total_images
overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

# Print the overall F1-Score
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1-Score: {overall_f1_score}")