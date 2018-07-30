import random
import os
import pickle
import json
import cv2
import numpy as np

def 切分(圖, l): 
    r,c=img.shape[:2]
    for x in range(0,r-l,l):
        for y in range(0,c-l,l):
            yield img[x:x+l,y:y+l]

def 讀圖(img_path):
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    if img is None:
        raise Exception('讀取失敗')
    if len(img.shape)==2:
        return img.reshape(*img.shape,1)
    return img[:,:,:3]

def 遍歷讀圖(路徑, 要名=False, 有限=False):
    while True:
        for root ,dirs ,files in os.walk(路徑):
            random.shuffle(files)
            for 名 in files:
                全名=os.path.join(root,名)
                圖 = 讀圖(全名)
                if 要名:
                    yield 圖,名 
                else:
                    yield 圖
        if 有限: 
            break

class 數據準備器:
    def __init__(self,數據源生成器,數據處理函數,輸出格式=None,擴充函數=None):
        self.數據源生成器=數據源生成器
        self.數據處理函數=數據處理函數
        self.輸出格式=輸出格式
        self.擴充函數=擴充函數

    def 重整格式(self,l):
        l=[np.array(i) for i in l]
        if self.輸出格式:
            a=zip(l,self.輸出格式)
            l=[np.reshape(d,t) for d,t in a]
        return l
    
    def 擴充生成器(self):
        for 源數據 in self.數據源生成器:
            if self.擴充函數:
                for 擴充數據 in self.擴充函數(源數據):
                    yield 擴充數據
            else:
                yield 源數據

    def 生成數據組(self,批大小=32):
        l=[[] for i in range(10)]
        for 數據 in self.生成數據():
            for i,x in enumerate(數據):
                l[i].append(x)
            if len(l[0])==批大小:
                yield self.重整格式(l[:self.數據個數])
                l=[[] for i in range(self.數據個數)]
    
    def 生成數據(self):
        while True:
            for 源數據 in self.擴充生成器():
                t=self.數據處理函數(源數據)
                self.數據個數=len(t)
                yield t