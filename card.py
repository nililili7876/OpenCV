import cv2
import numpy as np
from imutils import contours  # 排序操作，也可以不用。
 
 
# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
 
 
# 读取一个模板图像
img = cv2.imread('temp.png', )
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
 
# 轮廓检测
refCnts, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 排序(倒序操作) 得到正序0-9的轮廓
refCnts = sorted(refCnts, key=lambda b: b[0][0][0], reverse=False)
 
digits = {}
# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # 每一个数字对应每一个模板，此时模板中的10个数字分别被保存到了字典中
    digits[i] = roi
 
# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 读入银行卡
image = cv2.imread('./bankCard.png')
 
# 统一大小，这里建议让它变小一点，处理像素少一点，后面闭运算让其模糊也方便一些。
set_width = 300  # 自己设定 这里我统一了宽度
rate = set_width / image.shape[:2][1]
image = cv2.resize(image, (0, 0), fx=rate, fy=rate)
 
# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# 礼帽，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
 
# 梯度运算，这里使用Sobel算子，只进行了x方向计算。前面的礼帽操作是的我们梯度运算结果更干净些。
# ksize=-1相当于用3*3的
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
 
gradX = np.absolute(gradX)  # 绝对值，白-黑 黑-白
# 或者写为 cv2.convertScaleAbs(sobelx)
 
# 归一化处理
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX与最小值之间的距离占区间长度的几分之几
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
 
# 闭运算 把银行卡卡号那里弄模糊
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# 二值化，用于之后轮廓检测。
thresh = cv2.threshold(gradX, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
 
# 轮廓检测
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
 
locs = []
# 遍历轮廓
for (i, c) in enumerate(threshCnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
 
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.5 and ar < 4.0:
 
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))
 
# 将符合的这四组轮廓按x从左到右排序
locs = sorted(locs, key=lambda x: x[0])
 
# 遍历这四组数
output = []
 
# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
 
    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # 预处理
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 计算每一组的轮廓
    # group_,
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
 
    # 就是个排序 真正的顺序我们都知道，可以自己用sort函数
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
 
    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
 
        # 计算匹配得分
        scores = []
 
        # 在模板中计算每一个得分  字典digits记录了模板0-9
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
 
        # 得到最合适的数字，这里用的匹配方法对应得分越大越好。
        groupOutput.append(str(np.argmax(scores)))
 
    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 得到结果
    output.extend(groupOutput)
    
print("Credit Card #: {}".format("".join(output)))
cv_show('image',image)