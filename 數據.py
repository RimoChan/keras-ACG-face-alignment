import os
import random

import cv2
import numpy as np
import yaml

import koad

def 讀入(yml):
    with open(yml) as f:
        a=yaml.load(f)
    aa=list(a)
    while True:
        random.shuffle(aa)
        for i in aa:
            yield koad.讀圖('D:/數據集/yandere[370001~466745]/face_jpg/%s'%i),np.array(a[i])

def 轉圖(img,d):
    M = cv2.getRotationMatrix2D((128,128),d,1)
    img2 = cv2.warpAffine(img,M,(256,256), borderValue=[255,255,255])
    return img2
def 轉p(p,d):
    p=p.copy()
    x=p[:,0]-0.5
    y=p[:,1]-0.5
    d=d/180*np.pi
    xx=np.cos(d)*x+np.sin(d)*y
    yy=np.cos(d)*y-np.sin(d)*x
    p[:,0]=xx+0.5
    p[:,1]=yy+0.5
    return p

def 放大圖(img,d=1):
    M = cv2.getRotationMatrix2D((128,128),0,d)
    img2 = cv2.warpAffine(img,M,(256,256), borderValue=[255,255,255])
    return img2
def 放大p(p,d=1):
    p=p.copy()
    p=0.5+d*(p-0.5)
    return p

def 顛倒(p):
    p=p.copy()
    t=p[0].copy()
    p[0]=p[1].copy()
    p[1]=t
    for i in p:
        i[0]=1-i[0]
    return p

def 擴充(l):
    img,p=l
    img=img*(random.random()*0.30+0.8)
    img1=img[:,::-1]
    q=random.random()*0.2+1
    d=(random.random()-0.5)*2*12
    return (
        (轉圖(img ,d),轉p(p,d)),
        (轉圖(img1,d),轉p(顛倒(p),d)),
        (放大圖(img,q),放大p(p,q)),
        (放大圖(img1,q),放大p(顛倒(p),q)),
    )

class 數據準備器(koad.數據準備器):
    def __init__(self):
        super().__init__(
            數據源生成器 = 讀入(r"D:\數據集\yandere[370001~466745]\標註\練.yaml"),
            數據處理函數 = lambda t:  (t[0]/255,t[1]),
            擴充函數 = 擴充,
        )

class 測試準備器(koad.數據準備器):
    def __init__(self):
        super().__init__(
            數據源生成器 = 讀入(r"D:\數據集\yandere[370001~466745]\標註\測.yaml"),
            數據處理函數 = lambda t:  (t[0]/255,t[1]),
        )

def 畫(img,p):
    p=(p*256).astype(int)
    cv2.circle(img, (p[0][0],p[0][1]), 6, (1,0,0), 3)
    cv2.circle(img, (p[1][0],p[1][1]), 6, (0,1,0), 3)
    cv2.circle(img, (p[2][0],p[2][1]), 6, (0,0,1), 3)
    return img

if __name__=='__main__':
    for img, p in 測試準備器().生成數據():
    # for img, p in 數據準備器().生成數據():
        cv2.imshow('img',畫(img,p))
        cv2.waitKey(0)
