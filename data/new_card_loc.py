import cv2
import numpy as np
import math
import sys
# ACE算法
def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)
g_para = {}
def getPara(radius=5):  # 根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m
def zmIce(I, ratio=3, radius=300):  # 常规的ACE实现
    para = getPara(radius)
    height, width = I.shape

    # list1 = [[0] * radius + [x for x in range(height)] + [height - 1] * radius, [0] * radius + [x for x in range(width)] + [width - 1] * radius]
    # print(list1[0],list1[1])
    # zh = list1[0]
    # zw = list1[1]
    zh, zw = [0] * radius + [x for x in range(height)] + [height - 1] * radius, [0] * radius + [x for x in range(width)] + [width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res
def zmIceFast(I, ratio, radius):  # 单通道ACE快速增强实现
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5

    # print((width + 1) / 2,(height + 1) / 2)
    kw = int(((width + 1) / 2))
    kh = int((height + 1) / 2)
    Rs = cv2.resize(I, (kw, kh))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)
def zmIceColor(I, ratio=4, radius=3):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res
def hisEqulColor(img):
    '''
        直方图均衡化 用来处理光照
    '''
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0]) #equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq=cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq

#标准霍夫线变换
def line_detection(edges,image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    # # cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)    #点的坐标必须是元组，不能是列表。
    return image

# 定义全局参数用来
rui_l = [5, 5.5, 6, 6.5]
bilateralFilter_n = [0, 1, 2] #双边滤波运行次数
blur_k = [3, 5]
blur_n = [0, 1, 2] # 低通滤波次数
absdiff_k = [5, 6, 7, 8]
morph_close_k = [9, 10, 11, 12, 13, 14, 15, 16]

img_cloc = cv2.imread('./test_images/1.jpeg')

rows,cols, bytesPerComponent = img_cloc.shape
img = img_cloc.copy()

# 高斯滤波
img_cloc = cv2.GaussianBlur(img_cloc, ksize=(3, 3), sigmaX=0, sigmaY=0)
# ACE
img_cloc = zmIceColor(img_cloc/255.0)*255
img_cloc = np.uint8(img_cloc)

# 直方图均衡化
img_cloc = hisEqulColor(img_cloc)

img_tc = img_cloc.copy()

# 低通滤波器
img_cloc = cv2.blur(img_cloc, (3, 3))
# 双边滤波
img_cloc = cv2.bilateralFilter(img_cloc, 1, 100, 10)

# # 锐化
kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]], np.float32)  # 锐化
img_cloc = cv2.filter2D(img_cloc, -1, kernel=kernel)

# 双边滤波
img_cloc = cv2.bilateralFilter(img_cloc, 5, 100, 30)

# # 灰度图
img_cloc = cv2.cvtColor(img_cloc, cv2.COLOR_BGR2GRAY)

# 获得边缘
element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dilate = cv2.dilate(img_cloc, element)
dilate = cv2.dilate(dilate, element)
dilate = cv2.dilate(dilate, element)

erode = cv2.erode(img_cloc, element)
erode = cv2.erode(erode, element)
# erode = cv2.erode(erode, element)

# 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv2.absdiff(dilate, erode)

# # # 自适应二值化
# # scharrxy = cv2.adaptiveThreshold(scharrxy,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
ret2,result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret2)

# 开运算
#定义结构元素
kernelk1 = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
opened = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernelk1)

cv2.imwrite("./opened.jpg",opened)
cv2.namedWindow("scharrxy_otsu",0);
cv2.resizeWindow("scharrxy_otsu", 640, 480);
cv2.imshow("scharrxy_otsu",opened)
cv2.waitKey(0)



# 截取
con_num_roi = 0
max_mj = 0
max_next_mj = 0
iiimage, contours, hier = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    # 计算面积
    area=cv2.contourArea(c)
    # 去掉小的
    if area> (rows*cols*0.3) and area < (rows*cols*0.95):
        #看看有没有符合条件的
        con_num_roi = con_num_roi + 1
        if area>max_mj:
            max_next_mj = max_mj
            max_mj = area

if con_num_roi == 0:
    # 调参数重新来
    '''
            # 定义全局参数用来
        rui_l = [5, 5.5, 6, 6.5]
        bilateralFilter_n = [0, 1, 2] #双边滤波运行次数
        blur_k = [3, 5]
        blur_n = [0, 1, 2] # 低通滤波次数
        absdiff_k = [5, 6, 7, 8]
        morph_close_k = [9, 10, 11, 12, 13, 14, 15, 16]
    '''
    # global rui_l
    # global bilateralFilter_n
    # global blur_k
    # global blur_n
    # global absdiff_k
    # global morph_close_k
    for blur_kn in blur_k:
        # 低通滤波器
        img_tc = cv2.blur(img_tc, (blur_kn, blur_kn))
        for sk_n in bilateralFilter_n:
            if sk_n ==0:
                img_tc = img_tc
            elif sk_n == 1:
                # 双边滤波
                img_tc = cv2.bilateralFilter(img_tc, 1, 100, 10)
            elif sk_n == 2:
                # 双边滤波
                img_tc = cv2.bilateralFilter(img_tc, 1, 100, 10)
                # 双边滤波
                img_tc = cv2.bilateralFilter(img_tc, 1, 100, 10)
            for rui_ln in rui_l:
                # # 锐化
                kernel = np.array([[0, -1, 0], [-1, rui_ln, -1], [0, -1, 0]], np.float32)  # 锐化
                img_tc = cv2.filter2D(img_tc, -1, kernel=kernel)

                # 双边滤波
                img_tc = cv2.bilateralFilter(img_tc, 5, 100, 30)

                # # 灰度图
                img_tc = cv2.cvtColor(img_tc, cv2.COLOR_BGR2GRAY)

                for absdiff_k_n in absdiff_k:
                    # 获得边缘
                    element = cv2.getStructuringElement(cv2.MORPH_RECT, (absdiff_k_n, absdiff_k_n))
                    dilate = cv2.dilate(img_tc, element)
                    dilate = cv2.dilate(dilate, element)
                    dilate = cv2.dilate(dilate, element)

                    erode = cv2.erode(img_tc, element)
                    erode = cv2.erode(erode, element)
                    # 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
                    result_tc = cv2.absdiff(dilate, erode)

                    # # # 自适应二值化
                    # # scharrxy = cv2.adaptiveThreshold(scharrxy,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
                    ret2, result_tc = cv2.threshold(result_tc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # print(ret2)
                    for morph_close_k_n in morph_close_k:
                        # 开运算
                        # 定义结构元素
                        kernelk1 = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close_k_n, morph_close_k_n))
                        opened_tc = cv2.morphologyEx(result_tc, cv2.MORPH_CLOSE, kernelk1)

                        tes_num = 0
                        contours, hier = cv2.findContours(opened_tc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in contours:
                            # 计算面积
                            area = cv2.contourArea(c)
                            # 去掉小的
                            if area > (rows * cols * 0.3) and area < (rows * cols * 0.95):
                                tes_num = tes_num + 1
                        if tes_num == 1:
                            for c in contours:
                                # 计算面积
                                area = cv2.contourArea(c)
                                # 去掉小的
                                if area > (rows * cols * 0.3) and area < (rows * cols * 0.95):
                                    # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
                                    # x，y 是矩阵左上点的坐标，w，h 是矩阵的宽和高
                                    x, y, w, h = cv2.boundingRect(c)
                                    # [x-x,y-y]
                                    cropImg = img[y:y + h, x:x + w]
                                    cv2.namedWindow("scharrxy_otsu", 0);
                                    cv2.resizeWindow("scharrxy_otsu", 640, 480);
                                    cv2.imshow("scharrxy_otsu", cropImg)
                                    cv2.waitKey(0)
                                    tes_num = 0
                                    break
                                    exit(code=0)

    con_num_roi = 0

elif con_num_roi == 1:
    # 直接截取出来
    for c in contours:
        # 计算面积
        area = cv2.contourArea(c)
        print("area " + str(area))
        # 去掉小的
        if area > (rows * cols * 0.3) and area < (rows * cols * 0.95):
            # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
            # x，y 是矩阵左上点的坐标，w，h 是矩阵的宽和高
            x, y, w, h = cv2.boundingRect(c)
            # [x-x,y-y]
            cropImg = img[y:y + h, x:x + w]
            # 去一下边缘
            # 作为下一步的输入

            cv2.namedWindow("scharrxy_otsu", 0);
            cv2.resizeWindow("scharrxy_otsu", 640, 480);
            cv2.imshow("scharrxy_otsu", cropImg)
            cv2.waitKey(0)
            # 最后处理
            con_num_roi = 0
elif con_num_roi > 1:
    for c in contours:
        # 计算面积
        area = cv2.contourArea(c)
        print("area " + str(area))
        # 去掉小的
        if area > (rows * cols * 0.3) and area < (rows * cols * 0.95):
            x, y, w, h = cv2.boundingRect(c)
            if (area == max_mj or area == max_next_mj) and ((w/h>0.65 and w/h <0.95) or (w/h>1.05 and w/h<1.35)):
                # [x-x,y-y]
                cropImg = img[y:y + h, x:x + w]
                # 去一下边缘
                # 作为下一步的输入

                cv2.namedWindow("scharrxy_otsu", 0);
                cv2.resizeWindow("scharrxy_otsu", 640, 480);
                cv2.imshow("scharrxy_otsu", cropImg)
                cv2.waitKey(0)
                # 最后处理
                con_num_roi = 0
                max_mj = 0
                max_next_mj = 0
    # 最后处理
    con_num_roi = 0
else:
    print("出现了不会出现的错误！")

