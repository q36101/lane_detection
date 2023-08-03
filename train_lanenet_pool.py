# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:06:02 2020

@author: mediacore
"""

"""
訓練lanenet模型
"""
import sys

sys.path.append('/home/mediacore/lane_detection')

import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf
import random

try:
    from cv2 import cv2
except ImportError:
    pass

from config import global_config
from lanenet_model import lanenet_merge_model_pool
from data_provider import lanenet_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[:2]
        th, tw = self.size

        out_images = []
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = h1 + th
        w2 = w1 + tw

        cropped_image = img_group.copy()
        for i, img in enumerate(cropped_image):
            cropped_image[i] = img[h1:h2, w1:w2, ...]

        return cropped_image

class GroupRandomCropRatio(object):
    def __init__(self, size_ratio):
        self.size_ratio = size_ratio

    def __call__(self, img_group):
        h, w = img_group[0].shape[:2]
        target_w = int(self.size_ratio[0] * w)
        target_h = int(self.size_ratio[1] * h)

        out_images = []
        h1 = random.randint(0, max(0, h - target_h))
        w1 = random.randint(0, max(0, w - target_w))
        h2 = h1 + target_h
        w2 = w1 + target_w

        cropped_image = img_group.copy()
        for i, img in enumerate(cropped_image):
            cropped_image[i] = img[h1:h2, w1:w2, ...]

        return cropped_image

class GroupCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[:2]
        th, tw = self.size

        out_images = []
        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = h1 + th
        w2 = w1 + tw

        cropped_image = img_group.copy()
        for i, img in enumerate(cropped_image):
            cropped_image[i] = img[h1:h2, w1:w2, ...]

        return cropped_image

class GroupRandomRotation(object):
    def __init__(self, degree_range):
        self.degree_range = degree_range

    def __call__(self, img_group):
        degree = random.uniform(self.degree_range[0], self.degree_range[1])
        h, w = img_group[0].shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

        rotated_image = img_group.copy()
        for i, img in enumerate(rotated_image):
            rotated_image[i] = cv2.warpAffine(img, rotation_matrix, (w, h))

        return rotated_image

class GroupRandomBlur(object):
    def __init__(self, apply_prob):
        self.apply_prob = apply_prob

    def __call__(self, img_group):
        blurred_image = img_group.copy()
        for i, img in enumerate(blurred_image):
            if random.random() < self.apply_prob:
                blurred_image[i] = cv2.GaussianBlur(img, (5, 5), random.uniform(1e-6, 0.6))

        return blurred_image

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net_flag', type=str, help='Which base net work to you')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()

def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {}; Trainable params: {}'.format(flops.total_float_ops/1000000000.0, params.total_parameters))


def train_net(net_flag, dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)             #呼叫其他檔的class時，只會跑那個檔的_init_部分
    # print('train_dataset',train_dataset)
    # random.shuffle(train_dataset) 
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32,
                                  shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH, 3],
                                  name='input_tensor')
    binary_label_tensor = tf.compat.v1.placeholder(dtype=tf.int64,
                                         shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT,
                                                CFG.TRAIN.IMG_WIDTH, 1],
                                         name='binary_input_label')
    instance_label_tensor = tf.compat.v1.placeholder(dtype=tf.float32,
                                           shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT,
                                                  CFG.TRAIN.IMG_WIDTH],
                                           name='instance_input_label')  
    
    phase = tf.compat.v1.placeholder(dtype=tf.string, shape=None, name='net_phase')
    
    net = lanenet_merge_model_pool.LaneNet(net_flag=net_flag, phase=phase)
    
    # calculate the loss
    
    compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label_tensor,instance_label=instance_label_tensor, name='lanenet_loss')

    total_loss = compute_ret['total_loss']
    binary_seg_loss = compute_ret['binary_seg_loss']    #二值分割loss
    disc_loss = compute_ret['discriminative_loss']      #實例分割loss   
    l2_loss = compute_ret['l2_loss']
    dice_loss = compute_ret['dice_loss'] 
    focal_loss = compute_ret['focal_loss']           
    
    pix_embedding = compute_ret['instance_seg_logits']  # = (1, 256, 512, 3)
    
    #visualize attention map
    atgen_1 = compute_ret['output_1']
    atgen_2 = compute_ret['output_2']
    atgen_3 = compute_ret['output_3']
    atgen_4 = compute_ret['output_4']
    atgen_5 = compute_ret['output_5']
    decode_binary = compute_ret['decode_binary']

    # calculate the accuracy
    out_logits = compute_ret['binary_seg_logits']       # = (1, 256, 512, 2) decode完的結果(預測每個pixel是屬於背景還是線?)
    out_logits = tf.nn.softmax(logits=out_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1)   #tf.argmax : 回傳輸入tensor裡的最大值的index
    out = tf.argmax(out_logits, axis=-1)
    out = tf.expand_dims(out, axis=-1)   #tf.expand_dims : 插入一個1到input的tensor裡面，這裡的1是插在input的後面(增維)

    idx = tf.where(tf.equal(binary_label_tensor, 1))     #tf.equal : binary_label_tensor裡的白色pixel(值為1)(車道線)將會回傳TRUE，其他回傳FALSE     tf.where : 回傳2-D tensor，row為TRUE元素的數量，column為TRUE元素的座標
    pix_cls_ret = tf.gather_nd(out, idx)      #抽出out裡的idx位置的值
    accuracy = tf.compat.v1.count_nonzero(pix_cls_ret)                                           #算有幾個pixel不是0
    accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))    #tf.cast : 轉換輸入的dtypes，這裡轉成int64(小數點無條件捨去)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                               5000, 0.96, staircase=True)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=
                                           learning_rate).minimize(loss=total_loss,
                                                                   var_list=tf.compat.v1.trainable_variables(),
                                                                   global_step=global_step)

    # Set tf saver(存weight)
    saver = tf.compat.v1.train.Saver()
    model_save_dir = 'model/culane_lanenet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'culane_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary(紀錄訓練過程，可在tensorboard查看) 在CMD輸入tensorboard --logdir=檔案位置
    tboard_save_path = 'tboard/culane_lanenet/{:s}'.format(net_flag)
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.compat.v1.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.compat.v1.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.compat.v1.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.compat.v1.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.compat.v1.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.compat.v1.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    train_instance_seg_loss_scalar = tf.compat.v1.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    val_instance_seg_loss_scalar = tf.compat.v1.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    train_l2_loss = tf.compat.v1.summary.scalar(name='train_l2_loss', tensor=l2_loss)
    val_l2_loss = tf.compat.v1.summary.scalar(name='val_l2_loss', tensor=l2_loss)
    train_dice_loss = tf.compat.v1.summary.scalar(name='train_dice_loss', tensor=dice_loss)
    val_dice_loss = tf.compat.v1.summary.scalar(name='val_dice_loss', tensor=dice_loss)
    train_focal_loss = tf.compat.v1.summary.scalar(name='train_focal_loss', tensor=focal_loss)
    val_focal_loss = tf.compat.v1.summary.scalar(name='val_focal_loss', tensor=focal_loss)
    learning_rate_scalar = tf.compat.v1.summary.scalar(name='learning_rate', tensor=learning_rate)
    # output_1_map = tf.compat.v1.summary.image(name='output_1', tensor=atgen_1)
    # output_2_map = tf.compat.v1.summary.image(name='output_2', tensor=atgen_2)
    # output_3_map = tf.compat.v1.summary.image(name='output_3', tensor=atgen_3)
    # output_4_map = tf.compat.v1.summary.image(name='output_4', tensor=atgen_4)
    # output_5_map = tf.compat.v1.summary.image(name='output_5', tensor=atgen_5)
    # output_6_map = tf.compat.v1.summary.image(name='output_6', tensor=atgen_6)
    
    decode_binary_map = tf.compat.v1.summary.image(name='decode_binary', tensor=decode_binary)
    
    train_merge_summary_op = tf.compat.v1.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar, train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar, train_l2_loss, train_dice_loss, train_focal_loss, decode_binary_map])#, output_1_map, output_2_map, output_3_map])#, output_4_map, output_5_map])
                                                                                           
    val_merge_summary_op = tf.compat.v1.summary.merge([val_accuracy_scalar, val_cost_scalar, 
                                             val_binary_seg_loss_scalar, val_instance_seg_loss_scalar, val_l2_loss,val_dice_loss, val_focal_loss, decode_binary_map])#, output_1_map, output_2_map, output_3_map])
                                             #, output_4_map, output_5_map])                                                                                          

    # Set sess configuration
    
#限制GPU使用    sess_config = tf.ConfigProto(device_count={'GPU': 1})
#    
#    sess_config = tf.ConfigProto(log_device_placement=True)
#    
#    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
#    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
#    sess_config.gpu_options.allocator_type = 'BFC'
#    
#    sess_config = tf.ConfigProto(allow_soft_placement=True)
#限制GPU使用    sess_config.gpu_options.allow_growth = True
        
    #sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    
    sess = tf.compat.v1.Session(config=sess_config)

    # total_parameters = 0                                                #算參數量(可用，但stats_graph也會順便計算，這邊就註解掉)
    # for variable in tf.compat.v1.trainable_variables():
    #     # shape is an array of tf.Dimension
    #     shape = variable.get_shape()
    #     # print(shape)
    #     # print(len(shape))
    #     variable_parameters = 1
    #     for dim in shape:
    #         # print(dim)
    #         variable_parameters *= dim.value
    #     # print(variable_parameters)
    #     total_parameters += variable_parameters
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!PARAMETERS :',total_parameters)     #算參數量
    
    stats_graph(sess.graph)  #算計算量
    
    coord = tf.train.Coordinator()
    thread = tf.compat.v1.train.start_queue_runners(sess, coord)
    
    summary_writer = tf.compat.v1.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.io.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        if weights_path is None:
            log.info('Training from scratch')

        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            
        

        # 加载预训练参数
        if net_flag == 'vgg':
        
            pretrained_weights = np.load(
                './data/vgg16.npy', allow_pickle=True,
                encoding='latin1').item()
            print('You are using vgg16 pre-trained model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            for vv in tf.compat.v1.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                print(weights_key, 'AAAAAAAAAAAAAAAAA')
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.compat.v1.assign(vv, weights)
                    sess.run(_op)
                    print(vv.name, 'load success')
                except Exception as e:
                    print(vv.name, 'load failed')
                    continue
        if net_flag == 'res':
        
            pretrained_weights = np.load(
                './data/resnet_50.npy', allow_pickle=True,
                encoding='latin1').item()
            print('You are using resnet50 pre-trained model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            for vv in tf.compat.v1.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.compat.v1.assign(vv, weights)
                    sess.run(_op)
                    print(vv.name, 'load success')
                except Exception as e:
                    print(vv.name, 'load failed')
                    continue
        # if net_flag == 'mv3':
        
        #     pretrained_weights = np.load(
        #         './data/mobilenet_v3_large.npy', allow_pickle=True,
        #         encoding='latin1').item()
        #     print('You are using mobilenet_v3_large pre-trained model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     for vv in tf.compat.v1.trainable_variables():
        #         weights_key = vv.name.split('/')[-3:-1]
        #         print(weights_key, 'AAAAAAAAAAAAAAAAA')                
        #         try: 
        #             weights = pretrained_weights[weights_key][0]
        #             _op = tf.compat.v1.assign(vv, weights)
        #             sess.run(_op)
        #             print(vv.name, 'load success')
        #         except Exception as e:
        #             print(vv.name, 'load failed')
        #             continue

        train_cost_time_mean = []
        val_cost_time_mean = []
        train_start_time=time.time
        count=0
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()
            # color_map=[
            #     cv2.COLORMAP_AUTUMN,
            #     cv2.COLORMAP_BONE,
            #     cv2.COLORMAP_JET ,
            #     cv2.COLORMAP_WINTER,
            #     cv2.COLORMAP_RAINBOW ,
            #     cv2.COLORMAP_OCEAN ,
            #     cv2.COLORMAP_SUMMER ,
            #     cv2.COLORMAP_SPRING ,
            #     cv2.COLORMAP_COOL,
            #     cv2.COLORMAP_HSV ,
            #     cv2.COLORMAP_PINK ,
            #     cv2.COLORMAP_HOT ,
            #     cv2.COLORMAP_PARULA ,
            #     cv2.COLORMAP_MAGMA ,
            #     cv2.COLORMAP_INFERNO ,
            #     cv2.COLORMAP_PLASMA ,
            #     cv2.COLORMAP_VIRIDIS ,
            #     cv2.COLORMAP_CIVIDIS ,
            #     cv2.COLORMAP_TWILIGHT,
            #     cv2.COLORMAP_TWILIGHT_SHIFTED,
            #     cv2.COLORMAP_TURBO ,
            #     cv2.COLORMAP_DEEPGREEN ,
            # ]

            gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(CFG.TRAIN.T)     #一次讀入2張train圖片的資訊
            
            # if random.random() > 0.75:    
            #     instance_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in instance_gt_labels]    
            #     binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]                                                
            #     image = np.concatenate((gt_imgs, binary_gt_labels, instance_gt_labels), axis=-1)     #有1/4的機率會做crop(將y軸上面沒線的切掉)來增加data數量
            #     x = random.randint(1, 512)
            #     y = random.randint(1, 336)
            #     cropped_image = [tmp[y:y+ 384, x:x+ 768] for tmp in image]
            #     resized_image = [cv2.resize(tmp,
            #                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
            #                                 dst=tmp,
            #                                 interpolation = cv2.INTER_NEAREST)
            #                         for tmp in cropped_image]
            #     resized_image = np.array(resized_image)
            #     gt_imgs = resized_image[:, :, :, 0:3]
            #     gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
            #     binary_gt_labels = resized_image[:, :, :, 3:4]
            #     instance_gt_labels = resized_image[:, :, :, 4]
#                
            if random.random() < 0.25:    
               instance_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in instance_gt_labels]    
               binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]                                                
               image = np.concatenate((gt_imgs, binary_gt_labels, instance_gt_labels), axis=-1)     #有1/4的機率會做crop(將y軸上面沒線的切掉)來增加data數量
               x = random.randint(1,274)#0.4
               y = random.randint(1,100)
               cropped_image = [tmp[y:y+490, x:x+1366] for tmp in image]
               resized_image = [cv2.resize(tmp,
                                           dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                           dst=tmp,
                                           interpolation = cv2.INTER_NEAREST)
                                for tmp in cropped_image]
               resized_image = np.array(resized_image)
               gt_imgs = resized_image[:, :, :, 0:3]
               gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
               binary_gt_labels = resized_image[:, :, :, 3:4]
               instance_gt_labels = resized_image[:, :, :, 4]
               print('crop_image') 
               
            elif  0.25 < random.random() < 0.5:
                instance_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in instance_gt_labels] 
                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]                                                  
                image = np.concatenate((gt_imgs, binary_gt_labels, instance_gt_labels), axis=-1)     #有1/4的機率會做水平翻轉來增加data數量
                flip_image = [cv2.flip(tmp, 1) for tmp in image]                              
                resized_image = [cv2.resize(tmp,
                                            dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                            dst=tmp,
                                            interpolation = cv2.INTER_NEAREST)
                                    for tmp in flip_image]
                resized_image = np.array(resized_image)
                gt_imgs = resized_image[:, :, :, 0:3]
                gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
                binary_gt_labels = resized_image[:, :, :, 3:4]
                instance_gt_labels = resized_image[:, :, :, 4]
                print('flip_image') 


            gt_imgs = [cv2.resize(np.asarray(tmp),
                                (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                interpolation=cv2.INTER_LINEAR)
                    for tmp in gt_imgs]
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]

            binary_gt_labels = [cv2.resize(np.asarray(tmp),
                                        (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                        interpolation=cv2.INTER_NEAREST)
                            for tmp in binary_gt_labels]
            binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]

            instance_gt_labels = [cv2.resize(np.asarray(tmp),
                                            (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                            interpolation=cv2.INTER_NEAREST)
                                for tmp in instance_gt_labels]   
                
            # gt_imgs = [cv2.resize(tmp,
            #                       dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),     # CFG.TRAIN.IMG_WIDTH可改成512 ; CFG.TRAIN.IMG_HEIGHT可改成256
            #                       dst=tmp,
            #                       interpolation=cv2.INTER_LINEAR)
            #            for tmp in gt_imgs]
            # gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
            # binary_gt_labels = [cv2.resize(tmp,
            #                                dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
            #                                dst=tmp,
            #                                interpolation=cv2.INTER_NEAREST)
            #                     for tmp in binary_gt_labels]
            # binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]
            # instance_gt_labels = [cv2.resize(tmp,
            #                                  dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
            #                                  dst=tmp,
            #                                  interpolation=cv2.INTER_NEAREST)
            #                       for tmp in instance_gt_labels]
            
            phase_train = 'train'
            ##instance_loss,
            _, c, train_accuracy, train_summary, binary_loss ,l2, dice, focal, embedding, binary_seg_img ,output_1,output_2,output_3,output_4,output_5= \
                sess.run([optimizer, total_loss,                                 #上面一行的變數名稱不能在其他地方出現，會ERROR
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          #disc_loss,
                          l2_loss,
                          dice_loss,
                          focal_loss,
                          pix_embedding,
                          out_logits_out,
                          atgen_1,
                          atgen_2,
                          atgen_3,
                          atgen_4,
                          atgen_5,
                        #   atgen_6,
                        #   atgen_x, 
                        #   atgen_y, 
                        #   atgen_v,
                        #   atgen_d1, 
                        #   atgen_r1,
                          ],
                         feed_dict={input_tensor: gt_imgs,binary_label_tensor: binary_gt_labels,instance_label_tensor: instance_gt_labels,phase: phase_train})
            # 新的learning rate
            if epoch  == 10000:
                print('learning_rate',learning_rate)
                learning_rate = learning_rate*0.8 # 新的learning rate
            elif epoch  == 50000:
                print('learning_rate',learning_rate)
                learning_rate = learning_rate*0.5 # 新的learning rate
            elif (epoch>100000 & epoch%100000) == 0:
                print('learning_rate', learning_rate)
                learning_rate = learning_rate * 0.1
                    # sess.run(tf.compat.v1.assign(learning_rate, new_learning_rate))
            # if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):      #若c、binary_loss、instance_loss為nan(出問題ㄌ)則跑這個if下面的
            if math.isnan(c) or math.isnan(binary_loss):
                log.error('cost is: {:.5f}'.format(c))
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                # log.error('instance cost is: {:.5f}'.format(instance_loss))
                log.error('dice cost is: {:.5f}'.format(dice))
                log.error('focal cost is: {:.5f}'.format(focal))
                cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('nan_instance_label.png', instance_gt_labels[0])
                cv2.imwrite('nan_binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite('nan_embedding.png', embedding[0])
                return
            if epoch % 100 == 0:
                cv2.imwrite('./overseed/image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('./overseed/binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite('./overseed/instance_label.png', instance_gt_labels[0]*100)
                cv2.imwrite('./overseed/binary_seg_img.png', binary_seg_img[0] * 255)
                cv2.imwrite('./overseed/embedding.png', embedding[0]*100)
                cv2.imwrite('./overseed/output_1.png', output_1[0]*100)
                cv2.imwrite('./overseed/output_2.png', output_2[0]*100)
                cv2.imwrite('./overseed/output_3.png', output_3[0]*100)
                cv2.imwrite('./overseed/output_4.png', output_4[0]*100)
                cv2.imwrite('./overseed/output_5.png', output_5[0]*100)
                # cv2.imwrite('./overseed/output_6.png', output_6[0]*100)
                # cv2.imwrite('./overseed/output_2v.png', output_2v[0]*100)
                # cv2.imwrite('./overseed/output_3v.png', output_3v[0]*100)
                # cv2.imwrite('./overseed/output_4v.png', output_4v[0]*100)
                # cv2.imwrite('./overseed/output_5v.png', output_5v[0]*100)
                # cv2.imwrite('./overseed/output_x.png', output_x[0]*100)
                # cv2.imwrite('./overseed/output_y.png', output_y[0]*100)
                # cv2.imwrite('./overseed/output_v.png', output_v[0]*200)
                # cv2.imwrite('./overseed/output_d1.png', output_d1[0]*100)
                # cv2.imwrite('./overseed/output_r1.png', output_r1[0]*100)
            if epoch % 1000 == 0:
                cv2.imwrite('./overseed/see/image '+str(count)+'.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('./overseed/see/binary_label '+str(count)+'.png', binary_gt_labels[0] * 255)
                cv2.imwrite('./overseed/see/instance_label '+str(count)+'.png', instance_gt_labels[0])
                cv2.imwrite('./overseed/see/binary_seg_img '+str(count)+'.png', binary_seg_img[0] * 255)
                cv2.imwrite('./overseed/see/embedding '+str(count)+'.png', embedding[0])
                cv2.imwrite('./overseed/see/embedding_color '+str(count)+'.png', embedding[0]*100)
                # cv2.imwrite('./overseed/see/output_0 '+str(count)+'.png', output_0[0]*200)
                cv2.imwrite('./overseed/see/output_1 '+str(count)+'.png', output_1[0]*100)
                cv2.imwrite('./overseed/see/output_2 '+str(count)+'.png', output_2[0]*100)
                cv2.imwrite('./overseed/see/output_3 '+str(count)+'.png', output_3[0]*100)
                cv2.imwrite('./overseed/see/output_4 '+str(count)+'.png', output_4[0]*100)
                cv2.imwrite('./overseed/see/output_5 '+str(count)+'.png', output_5[0]*100)
                # cv2.imwrite('./overseed/see/output_6 '+str(count)+'.png', output_6[0]*100)
                # cv2.imwrite('./overseed/see/output_2v '+str(count)+'.png', output_2v[0]*100)
                # cv2.imwrite('./overseed/see/output_3v '+str(count)+'.png', output_3v[0]*100)
                # cv2.imwrite('./overseed/see/output_4v '+str(count)+'.png', output_4v[0]*100)
                # cv2.imwrite('./overseed/see/output_5v '+str(count)+'.png', output_5v[0]*100)
                # cv2.imwrite('./overseed/see/output_x1 '+str(count)+'.png', output_x1[0]*200)
                # cv2.imwrite('./overseed/see/output_x2 '+str(count)+'.png', output_x2[0]*200)
                # cv2.imwrite('./overseed/see/output_x3 '+str(count)+'.png', output_x3[0]*200)
                # cv2.imwrite('./overseed/see/output_x '+str(count)+'.png', output_x[0]*200)
                # cv2.imwrite('./overseed/see/output_y '+str(count)+'.png', output_y[0]*200)
                # cv2.imwrite('./overseed/see/output_v '+str(count)+'.png', output_v[0]*200)
                # cv2.imwrite('./overseed/see/output_d1 '+str(count)+'.png', output_d1[0]*100)
                # cv2.imwrite('./overseed/see/output_r1 '+str(count)+'.png', output_r1[0]*100)
                count+=1

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)      #一次讀入2張val圖片的資訊
                                
            for tmp in gt_imgs_val:
                if tmp is None:
                    print("Not found ~~")

            gt_imgs_val = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs_val]
            gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
            binary_gt_labels_val = [cv2.resize(tmp,
                                               dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                               dst=tmp)
                                    for tmp in binary_gt_labels_val]
            binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
            instance_gt_labels_val = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            _, c_val, val_summary, val_accuracy, val_binary_seg_loss, val_instance_seg_loss, val_l2, val_dice, val_focal = \
                sess.run([optimizer, total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss, l2_loss, dice_loss, focal_loss],  #上面一行的變數名稱不能在其他地方出現，會ERROR
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})

            if epoch % 100 == 0:
                cv2.imwrite('./overseed/test_image.png', gt_imgs_val[0] + VGG_MEAN)

            summary_writer.add_summary(summary=val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                # log.info('Epoch: {:d} total_loss= {:6f} binary_seg= {:6f} instance_seg= {:6f} l2= {:6f} dice= {:6f} focal= {:6f} '
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg= {:6f} l2= {:6f} dice= {:6f} focal= {:6f} '
                         'accuracy= {:6f} mean_cost_time= {:5f}s '.
                        #  format(epoch + 1, c, binary_loss, instance_loss, l2, dice, focal, train_accuracy,
                        format(epoch + 1, c, binary_loss, l2, dice, focal, train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg= {:6f} '
                         'instance_seg= {:6f} l2= {:6f} dice= {:6f} focal= {:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_l2, val_dice, val_focal, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                
            coord.request_stop()    
            coord.join(thread)
            # print("\n---------------FINAL RESULTS-----------------") 
            # print("Results after last Epochs")
            # print("Train binary loss:     ",binary_loss)
            # print("Train instance loss:   ",instance_loss)
            # print("Train accuracy:        ",train_accuracy)
            # print("Total training loss    ",c)

            # print("Test binary loss:      ",val_binary_seg_loss)
            # print("Test instance loss:    ",val_instance_seg_loss)
            # print("Test accuracy:         ",val_accuracy)
            # print("Total test loss        ",c_val)

            # print("\n---------------------------------------------")
            # print("\nRuntime: " + str(int((time.time() - train_start_time) / 60)) + " minutes")
            # print("\n---------------------------------------------")
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.net_flag, args.dataset_dir, args.weights_path)
    
