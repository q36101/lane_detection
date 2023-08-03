import cv2 as cv
import numpy as np


#读取图片
src = cv.imread('D:/Users/mediacore/lane_detection/data/tusimple_test_image/1.png')
#高斯降噪
src = cv.resize(src, (1280, 720))
cv.imwrite('D:/Users/mediacore/lane_detection/data/tusimple_test_image/t.png',src)
src1 = cv.GaussianBlur(src,(5,5),0,0)
# cv.imshow('gaosi',src1)
#灰度处理
src2 = cv.cvtColor(src1,cv.COLOR_BGR2GRAY)
# cv.imshow('huidu',src2)
#边缘检测
lthrehlod = 50
hthrehlod =150
src3 = cv.Canny(src2,lthrehlod,hthrehlod)
# cv.imshow('bianyuan',src3)
#ROI划定区间,并将非此区间变成黑色
regin = np.array([[(0,660),(690,440),
(1200,700),(src.shape[1],660)]]) #为啥要两中括号？
mask = np.zeros_like(src3)
mask_color = 255   #src3图像的通道数是1，且是灰度图像，所以颜色值在0-255
cv.fillPoly(mask,regin,mask_color)
src4 = cv.bitwise_and(src3,mask)
# cv.imshow('bianyuan2',src4)

#利用霍夫变换原理找出上图中的像素点组成的直线，然后画出来
rho = 1
theta = np.pi/180
threhold =15
minlength = 40
maxlengthgap = 20
lines = cv.HoughLinesP(src4,rho,theta,threhold,np.array([]),minlength,maxlengthgap)
#画线
linecolor =[0,255,255]
linewidth = 4
src5 = cv.cvtColor(src4,cv.COLOR_GRAY2BGR) #转化为三通道的图像




# 优化处理
def choose_lines(lines, threhold):  # 过滤斜率差别较大的点
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, x2, y1, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)  # 平均斜率
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threhold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines


lefts =[]
rights =[]
leftlength=[]
rightlength=[]
for line  in lines:
    for x1,y1,x2,y2 in line:
        #cv.line(src5,(x1,y1),(x2,y2),linecolor,linewidth)
        #分左右车道
        k = (y2-y1)/(x2-x1)
        length= ((y2-y1)**2+(x2-x1)**2)**0.5#计算线段长度
        if k<0:
                lefts.append(line)
                leftlength.append(length)
        else:
                rights.append(line)
                rightlength.append(length)

# print(max(leftlength))
# print(max(rightlength))

if max(leftlength)>max(rightlength):
    text="The left-hand side is the solid line"
else:
    text="The right-hand side is the solid line"



def clac_edgepoints(points, ymin, ymax):  # 可以理解成找一条线的端点
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    k = np.polyfit(y, x, 1)
    func = np.poly1d(k)  # 方程是y关于x的函数，因为输入的ymin ymax。要求xmin，xmax

    xmin = int(func(ymin))
    xmax = int(func(ymax))

    return [(xmin, ymin), (xmax, ymax)]


good_leftlines = choose_lines(lefts, 0.1)  # 处理后的点
good_rightlines = choose_lines(rights, 0.1)

leftpoints = [(x1, y1) for left in good_leftlines for x1, y1, x2, y2 in left]
leftpoints = leftpoints + [(x2, y2) for left in good_leftlines for x1, y1, x2, y2 in left]
rightpoints = [(x1, y1) for right in good_rightlines for x1, y1, x2, y2 in right]
rightpoints = rightpoints + [(x2, y2) for right in good_rightlines for x1, y1, x2, y2 in right]

lefttop = clac_edgepoints(leftpoints, 500, src.shape[0])  # 要画左右车道线的端点
righttop = clac_edgepoints(rightpoints, 500, src.shape[0])

src6 = np.zeros_like(src5)

cv.line(src6, lefttop[0], lefttop[1], linecolor, linewidth)
cv.line(src6, righttop[0], righttop[1], linecolor, linewidth)

# cv.imshow('onlylane',src6)

#图像叠加
src7 = cv.addWeighted(src1,0.8,src6,1,0)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(src7,text,(100,100), font, 1,(255,255,255),2)
cv.imshow('Finally Image',src7)

cv.waitKey(0)
cv.destroyAllWindows()