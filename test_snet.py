import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as keras
from keras.layers.normalization import BatchNormalization


def test_snet(pretrained_weights = None, input_size = (160,160,3), num_classes=2):
    inputs = Input(input_size)
    conv_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    bn1 = BatchNormalization(axis=3)(conv_1)
    conv1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    bn2 = BatchNormalization(axis=3)(conv_2)
    conv2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv_3 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    bn3 = BatchNormalization(axis=3)(conv_3)
    conv3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)
    #conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool9)


    merge10 = concatenate([conv8,pool9], axis = 3)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    #conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    #ex_1 = UpSampling2D(size = (2,2))(conv10)

    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
    #conv11 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool10)
    merge11 = concatenate([conv7,pool10], axis = 3)
    conv11 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
    #conv11 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)

    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    #conv12 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool11)
    merge12 = concatenate([conv6,pool11], axis = 3)
    conv12 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge12)
    #conv12 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)

    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    conv13 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool12)
    #conv13 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)


    x_newfc = GlobalAveragePooling2D(dim_ordering='default', name='global_pool')(conv9)
    x_newfc = Dense(num_classes, activation=None, name='fc10')(x_newfc)

    model = Model(input = inputs, output = x_newfc)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model






























