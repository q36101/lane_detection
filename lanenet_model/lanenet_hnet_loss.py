#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 下午1:12
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_loss.py
# @IDE: PyCharm Community Edition
"""
實現LaneNet的HNet損失函數
"""
import tensorflow as tf


def hnet_loss(gt_pts, transformation_coeffcient, name):
    """
    
    :param gt_pts: 原始的標籤點對 [x, y, 1]
    :param transformation_coeffcient: 映射矩陣參數(6參數矩陣) [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        # 首先映射原始標籤點對
        transformation_coeffcient = tf.concat([transformation_coeffcient, [1.0]], axis=-1)
        H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]])
        H_shape = tf.constant([9])
        H = tf.scatter_nd(H_indices, transformation_coeffcient, H_shape)
        H = tf.reshape(H, shape=[3, 3])

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # 求解最小二乘二階多項式擬合參數矩陣
        Y = tf.transpose(pts_projects[1, :])
        X = tf.transpose(pts_projects[0, :])
        Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
        Y_stack = tf.stack([tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)),
                      tf.expand_dims(X, -1))
        # 利用二階多項式參數求解擬合位置並反算到原始投影空間計算損失
        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, pts_projects[2, :]], axis=1))
        x_transformation_back = tf.matmul(tf.matrix_inverse(H), preds)

        loss = tf.reduce_mean(tf.pow(gt_pts[0, :] - x_transformation_back[0, :], 2))

    return loss


def hnet_transformation(gt_pts, transformation_coeffcient, name):
    """

    :param gt_pts:
    :param transformation_coeffcient:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        # 首先映射原始標籤點對
        transformation_coeffcient = tf.concat([transformation_coeffcient, [1.0]], axis=-1)
        H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]])
        H_shape = tf.constant([9])
        H = tf.scatter_nd(H_indices, transformation_coeffcient, H_shape)
        H = tf.reshape(H, shape=[3, 3])

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # 求解最小二乘二階多項式擬合參數矩陣
        Y = tf.transpose(pts_projects[1, :])
        X = tf.transpose(pts_projects[0, :])
        Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
        Y_stack = tf.stack([tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)),
                      tf.expand_dims(X, -1))

        # 利用二階多項式參數求解擬合位置
        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, pts_projects[2, :]], axis=1))
        preds_fit = tf.stack([tf.squeeze(x_preds, -1), Y], axis=1)
        x_transformation_back = tf.matmul(tf.matrix_inverse(H), preds)

    return preds_fit


if __name__ == '__main__':
    gt_labels = tf.constant([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                             [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]],
                            dtype=tf.float32, shape=[2, 3, 3])
    transformation_coffecient = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=tf.float32, shape=[6])

    _loss = hnet_loss(gt_labels, transformation_coffecient, 'loss')

    _pred = hnet_transformation(gt_labels, transformation_coffecient, 'inference')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_val = sess.run(_loss)
        pred = sess.run(_pred)
        print(loss_val)
        print(pred)
