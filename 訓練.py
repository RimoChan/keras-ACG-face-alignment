import os
import sys

import numpy as np
import cv2
import keras
from keras.models import Sequential, load_model
from keras.layers import *
from keras.layers.local import *
from keras.callbacks import *

import 數據
import 模型

#——————————————————————
def 訓練(model=None):
    if not model:
        model=模型.製造模型()

    訓練器=數據.數據準備器().生成數據組(批大小=8)
    測試器=數據.測試準備器().生成數據組(批大小=8)
    回調=[
        ModelCheckpoint('圖.h5',save_weights_only=True,monitor='val_loss',save_best_only=True),
        ReduceLROnPlateau(monitor='loss', factor=0.90, patience=5),
    ]

    model.fit_generator(
        訓練器, 600, 
        epochs=999,
        callbacks=回調, 
        validation_data = 測試器,
        validation_steps = 200//8,
        )

if __name__=='__main__':
    model=模型.製造模型()
    if os.path.isfile('圖.h5'):
        print('接着訓練')
        model.load_weights('圖.h5')
    else:
        print('重新訓練……')
    訓練(model)