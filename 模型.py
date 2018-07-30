import numpy as np
from keras.models import *
from keras.layers import *
from keras.layers.local import *
from keras.utils import plot_model
from keras.optimizers import *
from keras.losses import *

from coord import CoordinateChannel2D

dr = 0.1

def BN():
    return BatchNormalization()

def relu():
    return PReLU(shared_axes=[1,2])

def mult_res(inputs,n=2,bn=True):
    x=inputs
    for i in range(n):
        x=res(x,bn)
    return x

def res(inputs,bn=True):
    n=int(inputs.shape[-1])
    
    x=CoordinateChannel2D()(inputs)
    # x=(inputs)

    c1=relu()(Conv2D(n,3, padding='same')(x))
    c2=relu()(Conv2D(n,3, padding='same')(c1))
    m=add([c2,inputs])
    if bn:
        m=BN()(m)
    return m

def down(t,inputs,bn=True):
    c1=(relu()(Conv2D(t,5,strides=2, padding='same')(inputs)))
    return (mult_res(c1,bn=bn))

def up(t,inputs,bn=True):
    c1=(relu()(Conv2DTranspose(t,5,strides=2, padding='same')(inputs)))
    return (mult_res(c1,bn=bn))


def 製造模型():
    n=18

    inputs = Input(shape=[256,256,3])
    
    c128 = down(1 *n,inputs)
    c64 =  down(2 *n,c128)
    c32 =  down(4 *n,c64)
    c16 =  down(8 *n,c32)
    c8 =   down(16*n,c16)
    c4 =   down(32*n,c8)

    avg = GlobalAveragePooling2D()(c4)

    prediction = Reshape([3,2])(Dense(6,activation='sigmoid')(avg))

    model = Model(inputs=inputs, outputs=prediction)

    adam = Adam(lr=1e-3,amsgrad=True)
    model.summary()
    model.compile(loss='mse',optimizer=adam)
    
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    return model

if __name__=='__main__':
    製造模型()