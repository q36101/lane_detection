import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tusimple_evalu import LaneEval
# %matplotlib inline
# print('aaaaa')
json_pred = [json.loads(line) for line in open('C:/Users/HLY/Desktop/pre(5-22)/pool_add+dice_l1_focal_loss_vaildistest/pool_add+dice_l1_focal_loss_vaildistest.json').readlines()]
json_gt = [json.loads(line) for line in open('C:/Users/HLY/Downloads/test_set/test_label.json')]
# print('aaaaa')
pred, gt = json_pred[0], json_gt[0]
pred_lanes = pred['lanes']
run_time = pred['run_time']
gt_lanes = gt['lanes']
y_samples = gt['h_samples']
raw_file = gt['raw_file']

img = plt.imread(raw_file)
# plt.imshow(img)
# plt.show()

gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    for pt in lane:
        cv2.circle(img_vis, pt, radius=5, color=(0, 255, 0))

# plt.imshow(img_vis)
# plt.show()

gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=5)
for lane in pred_lanes_vis:
    cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=2)

# plt.imshow(img_vis)
# plt.show()

np.random.shuffle(pred_lanes)
# Overall Accuracy, False Positive Rate, False Negative Rate
# print LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
print(LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time))