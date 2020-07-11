## -*- coding: utf-8 -*-
from tkinter import *

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img

lst = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear','contempt','unknown','NF']
#lst = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear','contempt','unknown']
def onehot_to_label(res):
    label = ''
    for i in range(len(res[0])):
        if res[0][i] == 1:
            label = lst[i]
    return label


def softmax_to_label(res):
    label = ''
    #print(res)
    index = res[0].argmax()
    #print(index)
    label = lst[index]
    return label


model = load_model('K://DATA//python//face//model/model_v1.3.h5')


def predict(image):
    image = image.resize((48, 48), Image.NEAREST)  # 图像整体缩放到指定大小
    '''
    最近相邻插值算法/速度快/精度低的图像像素模拟方法
    取待采样点周围4个相邻像素点距离最近的1个邻点的灰度值作为该点的灰度值
    缩放图片时，缺少的像素通过直接照搬旁边的像素
    '''
    image = img_to_array(image)  # 数组
    image = image.reshape([-1, 48, 48, 1])  # reshape到4维张量
    res = model.predict(image)
    label = softmax_to_label(res)
    # print(res)
    return label


def openvideo():
    cv2.namedWindow('cam')
    cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier('K:/haarcascade_frontalface_alt2.xml')#Opencv级联分类器  haar人脸特征分类器 核心_Haar-like积分图计算
    '''
    基于机器学习,使用大量的正负样本训练得到分类器
    '''
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        '''#stageType、featureType、height、width、stageParams、featureParams、stages、features
        gray输入图像
        scaleFactor每次图像尺寸减小的比例
        minNeighbors每一个目标至少要被检测到3次（检测有效点数）
        minSize目标的最小尺寸

        '''
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 可能检测到多个人脸，循环单独框出每一张人脸
                try:
                    x, y, w, h = faceRect  # 获取框的左上的坐标，框的长宽
                    face = gray[x:x + w, y:y + h]
                    face = cv2.resize(face, (48, 48))
                    face = np.expand_dims(face, -1)
                    cv2.rectangle(frame, (x - 15, y - 15), (x + w - 15, y + h - 15), (0, 225, 0), 2)  # 外扩15个像素 划线rgb颜色  宽度
                    cv2.putText(frame, predict(array_to_img(face)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 230, 250), 1, cv2.LINE_AA)#抗锯齿线
                except:
                    continue
        try:
            cv2.imshow('cam', frame)
        except:
            continue
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):  # 按键q后break ord()将字符转化为对应的整数
            break
        '''
        On some systems, waitKey() may return a value that encodes more than just the ASCII keycode. 
        (A bug is known to occur on Linux when OpenCV uses GTK as its backend GUI library.)
        '''
    # 释放资源
    cap.release()
    cv2.destroyWindow('cam')
    print("cam off")
#print(predict(array_to_img(cv2.imread('K:/DATA/python/face/test.png'))))
print(predict(array_to_img(cv2.imread('K:/DATA/python/face/test4.png'))))
openvideo()
