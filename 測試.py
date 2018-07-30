import os
import sys

import yaml
import numpy as np
import cv2
from skimage import transform

import koad
import 模型
import 數據

model=模型.製造模型()
model.load_weights('圖.h5')

# 左眼，右眼，嘴
dst=np.array([
    [76, 110],
    [180, 110],
    [128, 195],
])
def 對正(圖,p):
    tform = transform.SimilarityTransform()
    tform.estimate(p*256, dst)
    M = tform.params[0:2, :]
    return cv2.warpAffine(圖, M, (256, 256), borderValue=[255,255,255])

def 得點(圖):
    global model
    return model.predict(圖.reshape([1,256,256,3])) [0]

for 圖 in koad.遍歷讀圖(r'D:\數據集\pixiv\つちくれ_face'):
    圖 = 圖/255
    p = 得點(圖)
    # 數據.畫(圖,p)
    圖2 = 對正(圖,p)
    cv2.imshow('圖',np.concatenate([圖,圖2],axis=1))
    cv2.waitKey()