# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:17:38 2020

@author: mediacore
"""

"""
測試LaneNet模型  
"""
import sys
sys.path.append('/home/mediacore/lane_detection') #path to fpn-lane-detection
import os
import os.path as ops
import argparse
import time
# import maths

import tensorflow as tf
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
#from lanenet_model import lanenet_merge_model_gru as lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_cluster_一階least畫線
from lanenet_model import lanenet_cluster_一階least畫線和補線
#from lanenet_model import lanenet_cluster_斜率補線
from lanenet_model import lanenet_postprocess
from config import global_config
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
def midpoint(ptA,ptB):
     return ((ptA[0]+ptB[0])*0.5,(ptA[1]+ptB[1])*0.5)

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()#先創造 argparse.ArgumentParser() 物件，它是我們管理引數需要的 “parser”
    
    parser.add_argument('--video_path', type=str, help='The vidoe path')#add_argument() 告訴 parser 我們需要的命令列引數有哪些
    parser.add_argument('--weights_path', type=str, help='The model weights path')#type=str:要求使用者輸入引數「必須」是 str 型別
    parser.add_argument('--net_flag', type=str, help='The model you use to train')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()#使用 parse_args() 來從 parser 取得引數傳來的 Data

def test_lanenet(video_path, weights_path, net_flag, use_gpu):

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.T, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3], name='input_tensor')#为将始终输入的张量插入占位符
    #dtype：张量中要输入的元素的类型。
    #shape：要输入的张量的形状（可选）。 如果未指定形状，则可以输入任何形状的张量。
    #name：操作的名称（可选）
    phase_tensor = tf.constant('train', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag=net_flag)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')

    videoCapture = cv2.VideoCapture(video_path) #影像擷取frame
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')#参數是MPEG-4编码類型，文件名后缀为.avi(MPEG-4是一套用於音訊、視訊資訊的壓縮編碼標準)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size=(180,220)

    videoWriter = cv2.VideoWriter('test2.mp4', fourcc, fps, size)#要將影片(攝影機影像還是影片)要存成圖片的話要使用cv2.imwrite()，如果想要存成影片檔的話，我們可以使用VideoWriter
    
    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()    
    saver = tf.compat.v1.train.Saver()#保存和恢復變量
    
    #控制GPU资源使用率 
    if use_gpu:
        sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})#参数 device_count，使用字典分配可用的 GPU 设备号和 CPU 设备号。
        print('You are using GPU!!!!!!!!!!!!!!!!')
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        print('You are not using GPU!!!!!!!!!!!!!!!!')
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION#程式最多能佔用指定的視訊記憶體
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH#<------------
    sess_config.gpu_options.allocator_type = 'BFC'#BFC算法

    sess = tf.compat.v1.Session(config=sess_config)
    
    frame_list = []
    count = 0
    time_predict = 0
    time_cluster = 0

    success, frame = videoCapture.read()#從攝影機擷取一張影像
    #print(type(videoCapture))
    
    t_start1 = time.time()
    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        f=0
        sum_big=0
        sum_small=0

        x_list=[]
        y_list=[]
        sum_a=0

        # Set sess configuration
        while success:
            # cv2.imshow('frame',frame)
            if (cv2.waitKey(1)& 0xFF == ord('q')):
             break
  

            log.info('開始讀取圖像數據並進行預處理')
            t_start = time.time()
            #image = frame
            #T=2適用，若T有改，這裡要再改        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            if count == 0:
                frame_list.append(frame)
                frame_list.append(frame)
                image_list = [cv2.resize(tmp, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR) for tmp in frame_list]
                image_merge_temp = image_list[1]
                image_list = [tmp - VGG_MEAN for tmp in image_list]
        
            else:
                frame_list.append(frame)
                frame_list_new = frame_list[count:count+2]
                image_list = [cv2.resize(tmp, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR) for tmp in frame_list_new]
                image_merge_temp = image_list[1]
                image_list = [tmp - VGG_MEAN for tmp in image_list]
            log.info('圖像讀取完畢, 耗時: {:.5f}s'.format(time.time() - t_start))#將浮點數舍入到小數點後5位
            
            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: image_list})
            t_detection_cost = time.time() - t_start
            log.info('單張圖像車道線預測耗時: {:.5f}s'.format(t_detection_cost))
            if(count == 0):
                time_predict = time_predict
            else:
                time_predict = time_predict + t_detection_cost

            t_start = time.time()
            binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])     #把有點歪的線刪掉，沒有這行線會比較多

            mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                        instance_seg_ret=instance_seg_image[0])

            test=binary_seg_image[0].astype(np.uint8)                    
            test=cv2.cvtColor(test,cv2.COLOR_GRAY2BGR)                    
                    
            # mask_image=mask_image*test

                                        
            # demo image
            # image_merge_temp = np.where((mask_image==0), image_merge_temp, mask_image)
            # image_merge_temp = cv2.resize(image_merge_temp, (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            #                                     int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            #                                     interpolation=cv2.INTER_LINEAR)
            t_post_process_cost = time.time() - t_start
            log.info('單張圖像車道線聚類耗時: {:.5f}s'.format(t_post_process_cost))
            if(count == 0):
                time_cluster = time_cluster
            else:
                time_cluster = time_cluster + t_post_process_cost
            h,w,c=test.shape
            # test=cv2.cvtColor(test,cv2.COLOR_GRAY2BGR)  
            test=cv2.dilate(test,None,iterations=2)
            mask_image=cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY) 
            # #############轉色##############
            # Label=mask_image
            # x,y=Label.shape
            # for m in range(x):
            #     for j in range(y):
            #         if Label[m,j].all()>0:
            #             Label[m,j]=255
            #         else:
            #             Label[m,j]=0
            # Label=cv2.cvtColor(Label,cv2.COLOR_GRAY2BGR) 

            #############轉色##############
            test=test*image_merge_temp

            # Label=Label*image_merge_temp
            # cv2.imshow('frame2', test)
            # cv2.waitKey(0)

            ###################ROI 區域##################
            img=test
            gray=cv2.GaussianBlur(img,(5,5),0)
            # cv2.imshow('frame2', gray)
            # cv2.waitKey(0)
            # gray=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) 
            # cv2.imshow('frame2', gray)
            # cv2.waitKey(0)
            edged=cv2.Canny(gray,70,200)
            # cv2.imshow('frame2', edged)
            # cv2.waitKey(0)
            gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame2', edged)
            # cv2.waitKey(0)
            

            edged=cv2.dilate(edged,None,iterations=2)
            
            edged=cv2.dilate(edged,None,iterations=2)
            # cv2.imshow('frame2', edged)
            # cv2.waitKey(0)
            w=int(img.shape[0])
            h=int(img.shape[1])
            c=int(img.shape[2])

            black = np.zeros((w,h,c),np.uint8)
            ############檢查roi位置##########
            # b1=black[int(img.shape[0]*18/20) : int(img.shape[0]*19/20) , int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]#int(img.shape[1]*1/40)
            # image_merge_temp[int(img.shape[0]*18/20) : int(img.shape[0]*19/20),int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]=b1
            # cv2.imshow('frame2',image_merge_temp)
            # cv2.waitKey(0)
            ############檢查roi位置##########



            img1=img[int(img.shape[0]*18/20) : int(img.shape[0]*19/20) , int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]#int(img.shape[1]*1/40)
            black[int(img.shape[0]*18/20) : int(img.shape[0]*19/20) , int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]=img1
            img1=black

            black = np.zeros((w,h,c),np.uint8)
            edged=edged[int(img.shape[0]*18/20) : int(img.shape[0]*19/20) , int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]
            edged=cv2.cvtColor(edged ,cv2.COLOR_GRAY2BGR)
            black[int(img.shape[0]*18/20) : int(img.shape[0]*19/20) , int(img.shape[1]*3/20) : int(img.shape[1]*17/20)]=edged
            edged=black
            edged=cv2.cvtColor(edged ,cv2.COLOR_BGR2GRAY)
            cnts,_=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.imshow('frame2', edged)
            # cv2.waitKey(0)

            # cv2.imshow('right',img1)

            order=1
            l=0
            line=0

            
            for c in cnts:    
                if cv2.contourArea(c) < 500:  
                    continue
                orig=img1.copy()
                box=cv2.minAreaRect(c)
                box=cv2.boxPoints(box)
                box=box.astype('int')
                line=img[int(np.min(box[:,1])) : int(np.max(box[:,1])) , int(np.min(box[:,0])) : int(np.max(box[:,0]))]
                # print(line.shape[0],line.shape[1])

                if (line.shape[0]!=0 | line.shape[1]!=0):
                    line=cv2.resize(line, (180,200),interpolation=cv2.INTER_LINEAR)
                    line = cv2.cvtColor(np.asarray(line), cv2.COLOR_RGB2BGR)
                    # 获得行数和列数即图片大小
                    rowNum, colNum = line.shape[:2]
                    sum=0
                    sum0=0
                    sum1=0
                    sum2=0
                    p=0
                    for x in range(0,rowNum,3):
                        for y in range(0,colNum,3):
                            # print(line[x,y].all())
                            if (line[x,y].all())>0:
                                p+=1
                                sum=sum+line[x,y]
                                sum0=sum0+line[x,y][0]
                                sum1=sum1+line[x,y][1]
                                sum2=sum2+line[x,y][2]
                                # print(line[x,y])
                                # print(line[x,y][0],line[x,y][1],line[x,y][2])
                            else:
                                sum=sum+0
                    if l==0:#右車道
                        # cv2.imshow('right',line)
                        right_sum=sum
                        print('rightsum',sum0,sum1,sum2)
                        #########頻率############
                        # if f%11==0:
                        #     start=time.time()   
                        # elif f%11==10:
                        #     waste=time.time()-start
                        #     print('waste',waste)
                        sum_a=sum0+sum1+sum2  
                        sum_a=int(sum_a/p/3)

                        # ###找最大值###
                        # if sum_a>=sum_big or f==0:
                        #     sum_big=sum_a
                        # else :
                        #     sum_big=sum_big
                        # ###找最大值###
                        
                        # ###找最小值###
                        # if sum_a<=sum_small or f==0 :
                        #     sum_small=sum_a
                        # else :
                        #     sum_small=sum_small
                        # ###找最小值###
                        
                        # cv2.waitKey(0)
                    else :#左車道
                        left_sum=sum
                        print('leftsum',sum0,sum1,sum2)
                        # cv2.imshow('left',line)
                        left_line=line
                        sum_b=sum0+sum1+sum2
                        # cv2.waitKey(0)


                    test=line
                else:
                    l=1
                (tl,tr,br,bl)=box
                (tltrX,tltrY)=midpoint(tl,tr)
                (tlblX,tlblY)=midpoint(tl,bl)
                (blbrX,blbrY)=midpoint(bl,br)
                (trbrX,trbrY)=midpoint(tr,br)
                x=int((tltrX+tlblX+blbrX+trbrX)/4)
                y=int((tltrY+tlblY+blbrY+trbrY)/4)
                
                # a=time.time()
                # b=a
                
                # print('111111111',b)

                # if b/5==0:
                #     if 500:
                #         cv2.putText(line,'solid',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #     else:
                #         cv2.putText(orig,'dash',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                # else:
                #         cv2.putText(orig,'',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                box=perspective.order_points(box)
                # cv2.drawContours(orig,[box.astype(int)],0,(0,255,0),4)  
                l+=1
                img1=orig
                order += 1
                print('llll',l)
            print('fffffffffffffffff',f)
            f+=1
            x_list.append(f)
            y_list.append(sum_a)
            



            # cv2.imshow('frame1', image_merge_temp)
            #print(image_merge_temp.shape)
            #mask image
            # test=binary_seg_image[0].astype(np.uint8)
            # test=cv2.cvtColor(test,cv2.COLOR_GRAY2BGR)
            # test = cv2.resize(test, (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))),interpolation=cv2.INTER_LINEAR)
            # mask_image = cv2.resize(mask_image, (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            #                                     int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            #                                     interpolation=cv2.INTER_LINEAR)
            #mask_image = cv2.resize(mask_image, (640,480),interpolation=cv2.INTER_LINEAR)
            # test=cv2.cvtColor(test,cv2.COLOR_GRAY2BGR) 
            # test=254*test
            # mask_image=mask_image*test
            # print(test.shape)
            
                                                
            # videoWriter.write(mask_image) #寫入影片
            cv2.imshow('frame2', line)
            # cv2.imwrite('C/Users/HLY/Desktop/see', line)
            videoWriter.write(test) #寫入影片
            
            success, frame = videoCapture.read() #讀取下一幀
            count += 1
        t_start2 = time.time()-t_start1
        plt.plot(x_list,y_list,'r-')
        plt.show()
        print('總預測車道線的部分平均每張耗時:',t_start2/count)
        print('預測車道線的部分平均每張耗時:', time_predict/count)
        print('進行車道線聚類的部分平均每張耗時:', time_cluster/count)
            
    sess.close()



if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(video_path=args.video_path, weights_path=args.weights_path, use_gpu=args.use_gpu, net_flag=args.net_flag)
      
        

# 選擇第二隻攝影機
#cap = cv2.VideoCapture(1)

#while(True):
  # 從攝影機擷取一張影像
  #ret, frame = cap.read()

  # 顯示圖片
  #cv2.imshow('frame', frame)

  # 若按下 q 鍵則離開迴圈
  #if cv2.waitKey(1) & 0xFF == ord('q'):
   # break

# 釋放攝影機
#cap.release()

# 關閉所有 OpenCV 視窗
#cv2.destroyAllWindows()
#mport cv2

