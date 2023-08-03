#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
實現LaneNet中實例分割的聚類部分
"""
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from config import global_config
import time
import warnings
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

CFG = global_config.cfg

class LaneNetCluster(object):
    """
    實例分割聚類器
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([0, 0, 255]),
                           np.array([0, 0, 255]),
                           np.array([0, 0, 255]),
                           np.array([0, 0, 255]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]   #BGR
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
#        print(labels.shape, 'labels')
#        print(cluster_centers.shape, 'cluster_centers')

        num_clusters = cluster_centers.shape[0]

        #log.info('聚類簇個數為: {:d}'.format(num_clusters))

        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):            #沒用到(另一個聚類方法)
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
        idx = np.where(binary_seg_ret == 1)      #求出2值分割圖值為1的pixel的index，2個3811*1

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])          #2值分割完屬於線的位子的點，根據這些位子訊息取出實例分割上的點(特徵向量)
            lane_coordinate.append([idx[0][i], idx[1][i]])
            
        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):          #沒用到
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
    def _lane_fit(lane_pts):           #沒用到
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
        #print(lane_coordinate, 'AAAAAAAAAAAAAAAAAAAAA')   #2值分割裡屬於線的pixel的座標

        num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1)       #原本bandwidth是設成1.5，線有時候會爆開

        # 聚類簇超過5個則選擇其中類內樣本最多的4個聚類簇保留下來
#        if num_clusters > 2:
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))     #屬於第i個類的有幾個點
            if cluster_sample_nums[i] < 100: #如果該類的點數太少(誤判為線)，則將誤判的刪掉不塗顏色
                del cluster_sample_nums[i]
                continue
            print(cluster_sample_nums, 'WWWWWWWWWWWWWWWWW')# = [1080, 1069, 959, 802, 7]  裡面數字為點數
            
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))   #按照大到小排列，會印出該數字的index而不是數字本身
        #print(sort_idx) = [0, 1, 2, 3, 4]
        cluster_index = np.array(range(num_clusters))[sort_idx[0:4]]      #取前4個
        #print(cluster_index, '!!!!!!!!!!!!!!!!!!!') = [0, 1, 2, 3]取前4個

#        else:           
#            cluster_index = range(num_clusters)

        mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)  #創一個(256,512,3)裡面數值都是0的array
        
        coord_list = []
        all_coord = []
        x_distance = []
        after_sort = []    #存原本每條線最下面(y都是256)的點離中心點的距離經過sort後會再x_distance_sort的哪個位置
        after_coord_list = []
        
        for index, i in enumerate(cluster_index):
            
            idx = np.where(labels == i)    #找出屬於第i類的pixel的index
#            print(idx[0], 'QQQQQQQQQQQQQQ')
            coord = lane_coordinate[idx]
#            coord = self._thresh_coord(coord)
            coord = np.flip(coord, axis=1)   #將axis=1這方向的值互換(X,Y座標互換)
#            print(coord, 'RRRRRRRRRRRRRRRRRRRRR')
            
            m_y = coord[0][1] - coord[len(coord)//2][1]# - coord[-1][1]  #一條線中間那個點和最後一個(圖最下面)的點的y座標相減
            m_x = coord[0][0] - coord[len(coord)//2][0]# - coord[-1][0]  #一條線中間那個點和最後一個(圖最下面)的點的x座標相減            
            m = (m_y/m_x)        #算每條的斜率
            print(m, 'mmmmmmmmmmmmmmmmm')
            #將線從線原本的底部補畫到圖的邊邊
            if(m > 0):                                         #2種情況，y2=256求x2 and x2=512求y2，求圖最邊邊且在線的斜率上的點
                if(m*(512-coord[-1][0])+coord[-1][1] < 256):   #線要延伸到圖的右邊，x2代512求y2
                    y2 = int(m*(512-coord[-1][0])+coord[-1][1])
#                    cv2.line(mask_image, (coord[-1][0], coord[-1][1]), (512, y2), (0, 0, 255), 2)
                    coord = np.concatenate((coord, np.array([[512, y2]])), axis=0)    #將求得的邊邊點跟原本線的點陣列結合(concate)
                    x_if_y256 = int(coord[-1][0]+((256-coord[-1][1])//m))    #當y座標是256(最下面)時算最右邊線的x座標(會超過512)
                    distance_rr = abs(x_if_y256-256)              #計算線(y值是256最下面)那個點跟圖中點(x值是256)的距離
                    x_distance.append(distance_rr)
                    
                else:                                          #線要延伸到圖的下面，y2代256求x2
                    x2 = int(coord[-1][0]+((256-coord[-1][1])//m))
#                    cv2.line(mask_image, (coord[-1][0], coord[-1][1]), (x2, 256), (0, 0, 255), 2)
                    coord = np.concatenate((coord, np.array([[x2, 256]])), axis=0)
                    distance_r = abs(x2-256)                     #計算線(y值是256最下面)那個點跟圖中點(x值是256)的距離
                    x_distance.append(distance_r)

            else:                                              #2種情況，y2=256求x2 and x2=0求y2，求圖最邊邊且在線的斜率上的點
                if(m*(0-coord[-1][0])+coord[-1][1] < 256):     #線要延伸到圖的左邊，x2代0求y2
                    y2 = int(m*(0-coord[-1][0])+coord[-1][1])
#                    cv2.line(mask_image, (coord[-1][0], coord[-1][1]), (0, y2), (0, 0, 255), 2)
                    coord = np.concatenate((coord, np.array([[0, y2]])), axis=0)
                    x_if_y256 = int(coord[-1][0]+((256-coord[-1][1])//m))   #當y座標是256(最下面)時算最左邊線的x座標(會是負的)
                    distance_ll = abs(x_if_y256-256)              #計算線(y值是256最下面)那個點跟圖中點(x值是256)的距離
                    x_distance.append(distance_ll)

                else:                                          #線要延伸到圖的下面，y2代256求x2
                    x2 = int(coord[-1][0]+((256-coord[-1][1])//m))
#                    cv2.line(mask_image, (coord[-1][0], coord[-1][1]), (x2, 256), (0, 0, 255), 2)
                    coord = np.concatenate((coord, np.array([[x2, 256]])), axis=0)
                    distance_l = abs(x2-256)                     #計算線(y值是256最下面)那個點跟圖中點(x值是256)的距離
                    x_distance.append(distance_l)       
            
            x_distance_sort = sorted(x_distance)    
            coord_list.append(coord)    #將屬於第i類的pixel座標存成一個list，有三條線list就有三個值
            
        for i in range(len(coord_list)):
            after_sort.append(x_distance.index(x_distance_sort[i]))   #找x_distance_sort的第0、1、2...個值在x_distance的哪個位置，再逐一放到after_sort裡，
            after_coord_list.append(coord_list[after_sort[i]])        #如此after_sort就代表x_distance裡由小到大的值的index，EX: x_distance=[665,221,220,644]，
#        print(after_sort, 'SSSSSSSSSSSSSSSS')                         #x_distance_sort=[220,221,644,665]，after_sort會=[2,1,3,0]，代表220在x_distance的第2位置...以此類推，
#        print(after_coord_list, 'AAAAAAAAAAAAAAA')                    #再根據after_sort的順序去排coord_list，這樣距離最近的線(220)會跑到after_coord_list的第一個，便可完成排序

            #coord = (coord[:, 0], coord[:, 1])

        for index in range(len(coord_list)):         #塗顏色        
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))  #抓取color_map裡的三個數字(BGR)
            
            cv2.polylines(img=mask_image, pts=[after_coord_list[index]], isClosed=False, color=color, thickness=2)       #在都是0的圖像中，利用判斷成是線的pixel的位子資訊，
                                                                                                                         #對這些位子的pixel上色
#        color = (int(self._color_map[index][0]),
#                 int(self._color_map[index][1]),
#                 int(self._color_map[index][2]))  #抓取color_map裡的三個數字(BGR) 
#        coord = np.array([coord])
#        cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)
        
#        all_coord = np.array(np.concatenate((after_coord_list[0], after_coord_list[1][::-1]), axis=0), np.int32)
#        print(coord_list[0], '!!!!!!!!!!!!!!')
#        print(coord_list[0][::-1], 'AAAAAAAAAAA')
#        cv2.fillPoly(mask_image, [all_coord], (0, 255, 0))    #塗綠色區域
            # mask_image[coord] = color

        return mask_image

if __name__ == '__main__':
    binary_seg_image = cv2.imread('C:/Users/mediacore/lane_detection/binary_ret.png', cv2.IMREAD_GRAYSCALE)
    binary_seg_image[np.where(binary_seg_image == 255)] = 1         #把255(白線)部分的值設為1
    instance_seg_image = cv2.imread('C:/Users/mediacore/lane_detection/instance_ret.png', cv2.IMREAD_UNCHANGED)
    ele_mex = np.max(instance_seg_image, axis=(0, 1))  
    
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)

    embedding_image = np.array(instance_seg_image, np.uint8)            #instance_seg_image跟embedding_image是一樣的(這行好像沒差)
    cluster = LaneNetCluster()
    mask_image = cluster.get_lane_mask(instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image)
    plt.figure('embedding')
    plt.imshow(embedding_image[:, :, (2, 1, 0)])   #最後(2,1,0)只是換顏色
    plt.figure('mask_image')
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.show()
