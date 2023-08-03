# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:45:40 2020

@author: mediacore

實現LaneNet中實例分割的聚類部分  lanenet_cluster_一階least畫線
"""
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time
import warnings
import cv2
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
        self._color_map = [#np.array([0, 0, 255]),   #BGR
                           #np.array([0, 255, 0]),
                           #np.array([255, 0, 0]),
                           #np.array([125, 125, 0]),
                           np.array([0, 0, 255]),
                           np.array([0, 0, 255]),
                           np.array([0, 0, 255]),
                           np.array([0, 0, 255]),
                           np.array([0, 0, 255]), #多加的
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
        ms = MeanShift(bandwidth, bin_seeding=True)
        # log.info('開始Mean shift聚類 ...')
        tic = time.time()
        try:
            ms.fit(prediction)
        except ValueError as err:
            log.error(err)
            return 0, [], []
        # log.info('Mean Shift耗時: {:.5f}s'.format(time.time() - tic))
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        # log.info('聚類簇個數為: {:d}'.format(num_clusters))

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
    def _thresh_coord(coord):
        """
        過濾實例車道線位置坐標點,假設車道線是連續的, 因此車道線點的坐標變換應該是平滑變化的不應該出現跳變
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

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

        num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

        # 聚類簇超過八個則選擇其中類內樣本最多的八個聚類簇保留下來
        if num_clusters > 8:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
                if cluster_sample_nums[i] < 100: #如果該類的點數太少(誤判為線)，則將誤判的刪掉不塗顏色
                    del cluster_sample_nums[i]
                    continue
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
        else:
            cluster_index = range(num_clusters)

        mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)
        model=LinearRegression()
        poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())  #7數字大小代表線越貼近點(點越分散會越彎曲)

        for index, i in enumerate(cluster_index):
            coord_x = []
            coord_y = []
            coordd = []
            idx = np.where(labels == i)
            coord = lane_coordinate[idx]
#            print(coord.shape[0], '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # coord = self._thresh_coord(coord)
            coord = np.flip(coord, axis=1)
            # coord = (coord[:, 0], coord[:, 1])
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))
                
            
#            print(coord[0][0][1], '!!!!!!')   #coord[0][0][0] 代表X的座標 [0][0][1] 代表Y的座標
#            print(coord[0][0])                  #coord[0][0]代表第一個點 [0][1]代表第二個點
#            print(len(coord[0]))                     #coord[0]代表該條線所有點
            for j in range(len(coord)):
                coord_x.append(coord[j][0])
                coord_y.append(coord[j][1])
                
            coord_x = np.array(coord_x)
            coord_y = np.array(coord_y)
            
            model.fit(coord_x[:,np.newaxis],coord_y)
            # poly_model.fit(coord_x[:, np.newaxis], coord_y)
            # coord_y = poly_model.predict(coord_x[:, np.newaxis])

            for k in range(len(coord)):         
                #coordd.append([coord_x[k], coord_y[k]])
                coordd.append([coord_x[k], int(model.intercept_+model.coef_*coord_x[k])])  #座標為(x, y=ax+b)，a為model.coef_，b為model.intercept_
            coordd = np.array([coordd])
            
            #coord = np.array([coordd])
            
            
            #model.fit(coord[0][0][:,np.newaxis],coord[0][1])
            
            cv2.polylines(img=mask_image, pts=coordd, isClosed=False, color=color, thickness=2)
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
