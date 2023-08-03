#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import sys
sys.path.append('/home/ycw/anaconda2/fpn-lane-detection')
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

from sklearn.manifold import TSNE

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--net_flag', type=str, help='The model you use to train')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
#    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def test_lanenet(image_path, weights_path, net_flag, use_gpu):
    """

    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('開始讀取圖像數據並進行預處理')
    t_start = time.time()
    frame_list = []
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    frame_list.append(image)
    frame_list.append(image)
    image_vis = image
    image_list = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR) for tmp in frame_list]
    image_merge_temp = image
    image_list = [tmp - VGG_MEAN for tmp in image_list]
    log.info('圖像讀取完畢, 耗時: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.T, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('train', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag=net_flag)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.compat.v1.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: image_list})
        t_cost = time.time() - t_start
        log.info('單張圖像車道線預測耗時: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])
        # mask_image = cluster.get_lane_mask_v2(instance_seg_ret=instance_seg_image[0])
        # mask_image = cv2.resize(mask_image, (image_vis.shape[1], image_vis.shape[0]),
        #                         interpolation=cv2.INTER_LINEAR)


        ele_mex = np.max(instance_seg_image[0], axis=(0, 1))
        for i in range(3):
            if ele_mex[i] == 0:
                scale = 1
            else:
                scale = 255 / ele_mex[i]
            instance_seg_image[0][:, :, i] *= int(scale)
        embedding_image = np.array(instance_seg_image[0], np.uint8)
        # cv2.imwrite('embedding_mask.png', embedding_image)

        # mask_image = cluster.get_lane_mask_v2(instance_seg_ret=embedding_image)
        # mask_image = cv2.resize(mask_image, (image_vis.shape[1], image_vis.shape[0]),
        #                         interpolation=cv2.INTER_LINEAR)

        #T-SNE
        w_image = instance_seg_image[0].shape[0]
        h_image = instance_seg_image[0].shape[1]
        c_image = instance_seg_image[0].shape[2]
        instance_seg_image_flatten = np.array(instance_seg_image[0]).reshape([w_image * h_image, c_image])
        

        #Ground Truth Path 請輸入該影像之instance ground truth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ground_truth_path = "C:/Users/mediacore/lane_detection/data/215_instance.png"
        img_ground_truth = cv2.imread(ground_truth_path , 0)
        img_ground_truth = cv2.resize(img_ground_truth, (512, 256), interpolation = cv2.INTER_CUBIC)
        w_ground_truth = img_ground_truth.shape[0]
        h_ground_truth = img_ground_truth.shape[1]

        img_ground_truth_flatten = np.array(img_ground_truth).reshape([w_ground_truth * h_ground_truth])

        line_index = img_ground_truth_flatten != 0
        line_img_ground_truth_flatten = img_ground_truth_flatten[line_index]
        line_instance_seg_image_flatten = instance_seg_image_flatten[line_index]

        tsne_embedded = TSNE(n_components=2, n_iter= 1000).fit_transform(line_instance_seg_image_flatten)

        vis_x = tsne_embedded[:, 0]
        vis_y = tsne_embedded[:, 1]

        cv2.imwrite('binary_ret.png', binary_seg_image[0] * 255)
        cv2.imwrite('instance_ret.png', embedding_image)
        cv2.imwrite('mask_image.png', mask_image)
        
        for i in range(511):
            for j in range(255):
                pixel_1 = mask_image[j,i]
                pixel_2 = image_merge_temp[j,i]
                if pixel_1[0]!=0 or pixel_1[1]!=0 or pixel_1[2]!=0:
                    pixel_2[0] = pixel_1[0]
                    pixel_2[1] = pixel_1[1]
                    pixel_2[2] = pixel_1[2]
                    image_merge_temp[j,i] = pixel_2
        cv2.imwrite('merge.png', image_merge_temp)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.figure('T-SNE')
        plt.scatter(vis_x, vis_y, c=line_img_ground_truth_flatten, cmap=plt.cm.get_cmap("jet", 5), marker='.')
        plt.colorbar(ticks=range(5))
        plt.clim(-0.5, 4.9)
        plt.show()

    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, net_flag, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('開始獲取圖像文件路徑...')
    #image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
    #                  glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
    #                  glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    image_path_list = glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) 

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.T, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('train', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag=net_flag)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.compat.v1.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / CFG.TRAIN.T))

        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] 開始圖像讀取和預處理...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 預處理{:d}張圖像, 共耗時: {:.5f}s, 平均每張耗時: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 預測{:d}張圖像車道線, 共耗時: {:.5f}s, 平均每張耗時: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
                binary_seg_image = cv2.resize(binary_seg_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)
                    #cv2.imwrite(image_save_path, binary_seg_image)
                    # log.info('[Epoch:{:d}] Detection image {:s} complete'.format(epoch, image_name))
            log.info('[Epoch:{:d}] 進行{:d}張圖像車道線聚類, 共耗時: {:.5f}s, 平均每張耗時: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))
            #mask_image.save("fileout.png", "JPEG") 

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(image_path=args.image_path, weights_path=args.weights_path, use_gpu=args.use_gpu, net_flag=args.net_flag)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size, net_flag=args.net_flag)
