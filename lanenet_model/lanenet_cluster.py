#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
實現LaneNet中實例分割的聚類部分  lanenet_cluster_沒畫線沒補線
"""
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import time
import warnings
import cv2

from itertools import cycle

try:
    from cv2 import cv2
except ImportError:
    pass


class LaneNetCluster(object):
    """
    實例分割聚類器
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([0, 0, 255]),   #BGR
                           np.array([0, 255, 0]),
                           np.array([255, 0, 0]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        pass

    @staticmethod
    def _cluster(prediction, bandwidth):
        """
        實現論文SectionⅡ的cluster部分
        :param prediction:
        :param bandwidth:
        :return:
        """
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False,cluster_all=False)
        # log.info('開始Mean shift聚類 ...')
        tic = time.time()
        try:
            label = ms.fit_predict(prediction)
            #print(label, '!!!!!!!!!!!!!')  #每個點都會回傳自己是哪一類(0到3類這樣)，跟座標點個數一樣
        except ValueError as err:
            log.error(err)
            return 0, [], []
        # log.info('Mean Shift耗時: {:.5f}s'.format(time.time() - tic))
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        log.info('聚類簇個數為: {:d}'.format(num_clusters))
        ##############可以每張圖都畫一個聚類圖，但只有中間一根??#################
        # plt.figure(1)
        # plt.clf()
        
        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(num_clusters), colors):
        #     my_members = labels == k
        #     cluster_center = cluster_centers[k]
        #     plt.plot(prediction[my_members, 0], prediction[my_members, 1], col + '.')
        #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=14)
        # plt.title('Estimated number of clusters: %d' % num_clusters)
        # plt.show()
        ######################################################################

        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        db = DBSCAN(eps=0.7, min_samples=200).fit(prediction)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        unique_labels = [tmp for tmp in unique_labels if tmp != -1]
        log.info('聚類簇個數為: {:d}'.format(len(unique_labels)))

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        return num_clusters, db_labels, cluster_centers

    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        通過二值分割掩碼圖在實例分割圖上獲取所有車道線的特徵向量
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 1)

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])

        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):  #這裡的coord還是x,y顛倒的狀態
        """
        過濾實例車道線位置坐標點,假設車道線是連續的, 因此車道線點的坐標變換應該是平滑變化的不應該出現跳變
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]  #其實是對上下做檢查(y軸 1~256)
        pts_y = coord[0, :]
        mean_x = np.mean(pts_x)
        mean_y = np.mean(pts_y)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x/2) or np.where(np.abs(pts_y - mean_y) < mean_y/2)
        # print(coord[idx[0]], 'AAAAAAAAAAAA')

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        車道線多項式擬合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f1 = np.polyfit(x, y, 3)
                p1 = np.poly1d(f1)
                x_min = int(np.min(x))
                x_max = int(np.max(x))
                x_fit = []
                for i in range(x_min, x_max + 1):
                    x_fit.append(i)
                y_fit = p1(x_fit)
            except Warning as e:
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)

    def get_lane_mask(self, binary_seg_ret, instance_seg_ret):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, instance_seg_ret)
        #print(lane_embedding_feats, 'QQQQQQQQQQQQQQQQQQQQQ')
        #print(len(lane_coordinate), 'AAAAAAAAAAAAAAAAAAAAA')   #2值分割裡屬於線的pixel的座標，個數(list長度)跟上面的一樣

        num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=0.05)

        # 聚類簇超過八個則選擇其中類內樣本最多的八個聚類簇保留下來
        # if num_clusters > 4:
        #     cluster_sample_nums = []
        #     for i in range(num_clusters):
        #         cluster_sample_nums.append(len(np.where(labels == i)[0]))
        #     sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
        #     cluster_index = np.array(range(num_clusters))[sort_idx[0:4]]
        # else:
        #     cluster_index = range(num_clusters)
            
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))     #屬於第i個類的有幾個點
            if cluster_sample_nums[i] < 200: #如果該類的點數太少(誤判為線)，則將誤判的刪掉不塗顏色
                cluster_sample_nums[i] == 0
                continue
            
            print(cluster_sample_nums, 'WWWWWWWWWWWWWWWWW')# = [1080, 1069, 959, 802, 7]  裡面數字為點數
            
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))   #按照大到小排列，會印出該數字的index而不是數字本身
        #print(sort_idx) = [0, 1, 2, 3, 4]
        cluster_index = np.array(range(num_clusters))[sort_idx[0:5]]      #取前4個
        #print(cluster_index, '!!!!!!!!!!!!!!!!!!!') = [0, 1, 2, 3]取前4個

        mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)

        for index, i in enumerate(cluster_index):
            idx = np.where(labels == i)
            coord = lane_coordinate[idx]
#            print(coord.shape[0], '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            coord = self._thresh_coord(coord)
            coord = np.flip(coord, axis=1)
            # coord = (coord[:, 0], coord[:, 1])
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))
            coord = np.array([coord])
#            print(coord.shape)
            
            cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=1)
            # mask_image[coord] = color

        return mask_image

if __name__ == '__main__':
    binary_seg_image = cv2.imread('C:/Users/mediacore/lane_detection/binary_ret.png', cv2.IMREAD_GRAYSCALE)
    binary_seg_image[np.where(binary_seg_image == 255)] = 1
    instance_seg_image = cv2.imread('instance_ret.png', cv2.IMREAD_UNCHANGED)
    ele_mex = np.max(instance_seg_image, axis=(0, 1))
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)
    embedding_image = np.array(instance_seg_image, np.uint8)
    cluster = LaneNetCluster()
    mask_image = cluster.get_lane_mask(instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image)
    plt.figure('embedding')
    plt.imshow(embedding_image[:, :, (2, 1, 0)])
    plt.figure('mask_image')
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.show()
