import matplotlib.pyplot as plt       #Matplotlib 是一个 Python 的 2D绘图库
import pylab
import matplotlib.image as img
import numpy as np                     #NumPy是Python的一种开源的数值计算扩展
import cv2
import os
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

def show(name,img):
     cv2.imshow(name,img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
def midpoint(ptA,ptB):
     return ((ptA[0]+ptB[0])*0.5,(ptA[1]+ptB[1])*0.5)
i=0
a=0
while i<10000 :

    img=cv2.imread('D:/Users/mediacore/lane_detection/data/training_data_example/training_data/image_gt_binary/'+str(i)+'.png')

    # image=img.imread('D:/Users/mediacore/lane_detection/data/tusimple_200frame/1.png')
    #圖片預處理
    # img=cv2.imread('D:/Users/mediacore/lane_detection/testing2.png')
    width=25
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # mask=np.zeros(img.shape, np.uint8)
    # pts=[(0,0),(256,0),(256,512),(0,0)]
    # points = np.array(pts, np.int32)
    # print('points',points)
    # points = points.reshape((-1, 1, 2))
    # print('points',points)
    # mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    # print('mask',mask)
    # mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
    # print('mask2',mask2)
    # mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 用于 显示在桌面的图像
    # print('mask3',mask3)

    # cv2.imshow("mask", mask2)
    # cv2.waitKey(0)

    gray=cv2.GaussianBlur(img,(5,5),0)

    edged=cv2.Canny(gray,70,200)
    # cv2.imshow('name1',edged)
    # cv2.imwrite('./name1.png',edged)
    # cv2.waitKey(0)
    edged=cv2.dilate(edged,None,iterations=1)
    # cv2.imshow('name2',edged)
    # cv2.imwrite('./name2.png',edged)
    # cv2.waitKey(0)
    edged=cv2.erode(edged,None,iterations=1)
    # cv2.imshow('name3',edged)
    cv2.imwrite('./name3.png',edged)
    # cv2.waitKey(0)
    edged=cv2.dilate(edged,None,iterations=2)
    # edged=cv2.erode(edged,None,iterations=1)
    # edged=cv2.erode(edged,None,iterations=1)
    # print((int(img.shape[0]),int(img.shape[1]),int(img.shape[2])))
    # print(type((int(img.shape[0]),int(img.shape[1]),int(img.shape[2]))))
    w=int(img.shape[0])
    h=int(img.shape[1])
    c=int(img.shape[2])
    i+=1
    
    b=0

# print(type(w))

# size=((w,h,c),np.uint8)
# black = np.zeros(size)
    # img=img[:,:,]

    # black = np.zeros((w,h,c),np.uint8)
    # img1=img[int(img.shape[0]*3/5) : int(img.shape[0]*4/5) , 0 : int(img.shape[1])]
    # black[int(img.shape[0]*3/5) : int(img.shape[0]*4/5) , 0 : int(img.shape[1])]=img1
    # img1=black

    # black = np.zeros((w,h,c),np.uint8)
    # edged=edged[int(img.shape[0]*3/5) : int(img.shape[0]*4/5) , 0 : int(img.shape[1])]
    # edged=cv2.cvtColor(edged ,cv2.COLOR_GRAY2BGR)
    # black[int(img.shape[0]*3/5) : int(img.shape[0]*4/5) , 0 : int(img.shape[1])]=edged
    # edged=black
    # edged=cv2.cvtColor(edged ,cv2.COLOR_BGR2GRAY)


    img1=img[int(img.shape[0]*0/16): int(img.shape[0]*4/16) , int(img.shape[1]*0/16) : int(img.shape[1]*16/16)]
    rowNum, colNum = img1.shape[:2]
    for x in range(rowNum):
        for y in range(colNum):
            if  img1[x, y].tolist() == [255,255,255] :
                b+=1
    if b>1:                     
        a+=1
        show('a',img1)
    print(a)
    
                
    
    # show('a',img1)

    




# # cv2.imshow('name3',img1)
# # cv2.waitKey(0)
# # cv2.imshow('name3',edged)
# # cv2.waitKey(0)
# cnts,_=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# # print('cnts',cnts)
# # print('_',_)

# # cv2.drawContours(img,cnts,-1,(0,0,255),3)

# # (cnts,_)=contours.sort_contours(cnts)
# # cv2.drawContours(img,cnts,-1,(0,0,255),3)
# # pixelPerMetricX=0
# # pixelPerMetricY=0

# order=1
# l=0
# for c in cnts:
    
#     if cv2.contourArea(c) < 200:  
#         continue
#     orig=img1.copy()
#     box=cv2.minAreaRect(c)
#     box=cv2.boxPoints(box)

#     #############找點#############
#     left_point_x=np.min(box[:,0])
#     right_point_x=np.max(box[:,0])
#     top_point_y=np.min(box[:,1])
#     bottom_point_y=np.max(box[:,1])

#     # left_point_y=box[:,1][np.where(box[:,0]==left_point_x)][0]
#     # right_point_y=box[:,1][np.where(box[:,0]==right_point_x)][0]
#     # top_point_x=box[:,0][np.where(box[:,1]==top_point_y)][0]
#     # bottom_point_x=box[:,0][np.where(box[:,1]==bottom_point_y)][0]

#     # vertices=np.array([[top_point_x,top_point_y],[bottom_point_x,bottom_point_y],[left_point_x,left_point_y],[right_point_x,right_point_y]])/
#     # cv2.imshow('orig',orig)
#     # cv2.waitKey(0)
#     #############找點#############

#     box=box.astype('int')
#     # print('box',box)

#     ##################切出來#####################
#     # box_line=np.array([box])
#     # mask=np.zeros(img.shape[:2],np.uint8)
#     # cv2.polylines(mask,box_line,1,255)
#     # cv2.fillPoly(mask,box_line,255)

#     # dst=cv2.bitwise_and(img,img,mask=mask)
#     # cv2.imshow('dst',dst)
#     # cv2.waitKey(0)

#     # bg=np.ones_like(img,np.uint8)*255
#     # cv2.bitwise_not(bg,bg,mask=mask)
#     # cv2.imshow('bg',bg)
#     # cv2.waitKey(0)

#     # dst_white=bg+dst
#     # cv2.imshow('dst_white',dst_white)
#     # cv2.waitKey(0)
#     ##################切出來#####################
#     print(int(np.min(box[:,0])) , int(np.max(box[:,0])) , int(np.min(box[:,1])) , int(np.max(box[:,1])))

#     line=img[int(np.min(box[:,1])) : int(np.max(box[:,1])) , int(np.min(box[:,0])) : int(np.max(box[:,0]))]
    
#     if l==0:
#         cv2.imshow('right',line)
#         cv2.waitKey(0)
#     else :
#         cv2.imshow('left',line)
#         cv2.waitKey(0)
#     l+=1

#     line = cv2.cvtColor(np.asarray(line), cv2.COLOR_RGB2BGR)
#     # 获得行数和列数即图片大小
    
#     rowNum, colNum = line.shape[:2]
#     sum=0
#     sum0=0
#     sum1=0
#     sum2=0
#     for x in range(0,rowNum,3):
#         for y in range(0,colNum,3):
#             # print(line[x,y].all())
#             if (line[x,y].all())>0:
#                 sum=sum+line[x,y]
#                 sum0=sum0+line[x,y][0]
#                 sum1=sum1+line[x,y][1]
#                 sum2=sum2+line[x,y][2]
#                 # print(line[x,y])
#                 print(line[x,y][0],line[x,y][1],line[x,y][2])
#             else:
#                 sum=sum+0
#     print('sum',sum0,sum1,sum2)
#     print('shape',line.shape)

#     ####################################
#     # for x in range(0,rowNum,3):
#     #     for y in range(0,colNum,3):
#     #             line[x, y] = np.array([255,255,255])
#     # cv2.imshow('img',line)
#     # cv2.waitKey(0)
#     ####################################

        

   

#     # for i in range(x_min, x_max + 1):
#     #                 x_fit.append(i)
#     #             y_fit = p1(x_fit)

#     box=perspective.order_points(box)
#     cv2.drawContours(orig,[box.astype(int)],0,(0,255,0),4)  
    

    
#     # for x,y in box:

#         # cv2.circle(orig,(int(x),int(y)),5,(0,0,255),5) 
#         # print('red',x,y)
#         # cv2.imshow('frame',orig)
#         # cv2.waitKey(0)
        
#     (tl,tr,br,bl)=box
#     print((tl,tr,br,bl))
#     print(box)
#     (tltrX,tltrY)=midpoint(tl,tr)
#     (tlblX,tlblY)=midpoint(tl,bl)
#     (blbrX,blbrY)=midpoint(bl,br)
#     (trbrX,trbrY)=midpoint(tr,br)

#     # left=
#     # right=


#     # cv2.circle(orig,(int(x),int(y)),5,(0,0,255),5)
        
#     # print('red',x,y)
#     # cv2.imshow('frame',orig)
#     # cv2.waitKey(0)


#     # print((tltrX,tltrY)\
#     #     ,(tlblX,tlblY)\
#     #     ,(blbrX,blbrY)\
#     #     ,(trbrX,trbrY))
#     # cv2.circle(orig,(int(tltrX),int(tltrY)),5,(200,0,0),-1)
#     # cv2.circle(orig,(int(tlblX),int(tlblY)),5,(0,200,0),-1)
#     # cv2.circle(orig,(int(blbrX),int(blbrY)),5,(200,0,0),-1)
#     # cv2.circle(orig,(int(trbrX),int(trbrY)),5,(0,200,0),-1)



#     # cv2.line(orig,(int(tltrX),int(tltrY)),(int(blbrX),int(blbrY)),(255,0,0),2)
#     # cv2.line(orig,(int(blbrX),int(blbrY)),(int(trbrX),int(trbrY)),(255,0,0),2)

    

#     dA=dist.euclidean((tltrX,tltrY),(blbrX,blbrY))

#     mA=(tltrY-blbrY)/(tltrX-blbrX)
#     # print('mA',mA)

#     dB=dist.euclidean((tlblX,tlblY),(trbrX,trbrY))

#     MB=(tlblY-trbrY)/(tlblX-trbrX)

#     # print('MB',MB)

    

#     pts = np.array([(tlblX,tlblY),(trbrX,trbrY),(trbrX+mA*dA/10,trbrY*mA*dA/10),(tlblX+mA*dA/10,tlblY+mA*dA/10)])

#     # cv2.circle(orig,(int(tlblX),int(tlblY)),5,(0,255,0),-1)
#     # cv2.circle(orig,(int(trbrX),int(trbrY)),5,(0,255,0),-1)
#     # cv2.circle(orig,(int(trbrX+mA*dA*0.5),int(trbrY*mA*dA*0.5)),5,(0,255,0),-1)
#     # cv2.circle(orig,(int(tlblX+mA*dA*0.5),int(tlblY+mA*dA*0.5)),5,(0,255,0),-1)


#     # cv2.polylines(orig,[pts],True,(0,0,255),5)


#     # print('長 : ',dB)
#     # print('寬 : ',dA)
#     # if pixelPerMetricX ==0 or pixelPerMetricY ==0:
#     #      pixelPerMetricX = dB/width
#     #      pixelPerMetricY = dA/width
#     # dimA=dA/pixelPerMetricY
#     # dimB=dB/pixelPerMetricX
#     mylist = [tltrY,blbrY,tlblY,trbrY]
#     # print(max(mylist))
#     if max(mylist)==trbrY:
#         cv2.circle(orig,(int(trbrX),int(trbrY)),5,(255,0,0),-1)
#     elif max(mylist)==tltrY:
#         cv2.circle(orig,(int(tltrX),int(tltrY)),5,(255,0,0),-1)
#     elif max(mylist)==tlblY:
#         cv2.circle(orig,(int(tlblX),int(tlblY)),5,(255,0,0),-1)
#     else :
#         cv2.circle(orig,(int(blbrY),int(blbrY)),5,(255,0,0),-1)



#     x=int((tltrX+tlblX+blbrX+trbrX)/4)
#     y=int((tltrY+tlblY+blbrY+trbrY)/4)

#     ##############dash or solid#####################
#     # if dB >500:
#     #     cv2.putText(orig,'solid',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     # else:
#     #     cv2.putText(orig,'dash',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
#     # cv2.putText(orig,"{:.1f}nm".format(dimB),(int(tltrX)-10,int(tltrY)),cv2.FONT_HERSHEY_COMPLEX,0.1,(255,0,0),1)
#     # cv2.putText(orig,"{:.1f}nm".format(dimA),(int(trbrX)-10,int(trbrY)),cv2.FONT_HERSHEY_COMPLEX,0.1,(255,255,255),1)

#     # cv2.imwrite('1.jpg'.format(order),orig)
#     # cv2.imshow('frame',orig)
#     # cv2.waitKey(0)
#     # print(orig)
#     img1=orig
#     order += 1
#     # print(c)
#     # cv2.imshow('frame',orig)
# # print('111111111111111111',img.shape,orig.shape)
# print(int(img.shape[0]*3/5),int(img.shape[0]*4/5) , 0,int(img.shape[1]))
# cv2.imshow('frame',orig)
# cv2.waitKey(0)
# img[int(img.shape[0]*3/5) : int(img.shape[0]*4/5) , 0 : int(img.shape[1])] = orig[int(img.shape[0]*3/5) : int(img.shape[0]*4/5) , 0 : int(img.shape[1])]
# cv2.imshow('frame',orig)
# cv2.waitKey(0)
# cv2.imwrite('1.jpg',img)
# cv2.imshow('frame',img)
# cv2.waitKey(0)
