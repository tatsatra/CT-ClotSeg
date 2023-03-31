# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:17:06 2023

@author: munjalpu
"""

#CNN library, high level
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input, Dropout, Concatenate, UpSampling2D
from tensorflow.keras.regularizers import l1, l2, l1_l2 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import get_custom_objects
import matplotlib.pyplot as plt
from scipy.io import savemat
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import *
import keras
# from tensorflow.keras.utils.vis_utils import plot_model


path_image=''
model_checkpoint=''
best_model=''
l2_magnitude=0.0001

IMAGE_SHAPE = (256, 256, 2)

filter1 = 32
filter2 = 64
filter3 = 128
filter4 = 256


padding_val1="valid" 
padding_val2="same"

activation_val1="relu"
activation_val2="sigmoid"

IMAGE_ORDERING = 'channels_last'


epsilon_val=0.0001
stride_val=(2, 2)
filter_size=(3 , 3)
filter_size2=(1 , 1)
max_pool_size=(2 , 2)
up_scale_size= (2 , 2)
drop_frac1=0.1
drop_frac2=0.5


num_classes=10
LEARNING_RATE=0.0005


def custom_loss(y_true,y_pred):
    return K.mean(0.01*(1-y_true)*K.binary_crossentropy(y_true,y_pred)+0.99*y_true*K.binary_crossentropy(y_true,y_pred),axis=-1)

def unbalanced_loss_hashemi(y_true,y_pred):
    beta = 1.5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (1 + beta ** 2) * K.sum(y_true_f * y_pred_f) / ((1 + beta ** 2) * K.sum(y_true_f * y_pred_f) + (beta ** 2) * K.sum((1-y_pred_f)*y_true_f) + K.sum(y_pred_f * (1 - y_true_f)))

def dice_coef(y_true,y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)

def bce_dice_loss(y_true, y_pred):
    #return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return dice_coef_loss(y_true, y_pred)



def down(filters, input_,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,l2_magnitude):
    
    down_ = Convolution2D(filters, filter_size, kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(input_)
    down_ = BatchNormalization(epsilon=epsilon_val)(down_)
    down_ = Dropout(drop_frac1)(down_)
    
    down_ = Convolution2D(filters, filter_size, kernel_regularizer=l2(l2_magnitude), padding=padding_val2,  data_format=IMAGE_ORDERING)(down_)
    down_res = BatchNormalization(epsilon=epsilon_val)(down_)

    down_pool = MaxPooling2D(max_pool_size, strides=stride_val, data_format=IMAGE_ORDERING)(down_res)
    
    return down_pool, down_res

def up(filters, input_, down_high,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,up_scale_size,l2_magnitude):
    
    up_ = UpSampling2D(up_scale_size, data_format=IMAGE_ORDERING)(input_)
    up_ = Concatenate(axis=3)([down_high, up_])
    
    up_ = Convolution2D(filters, filter_size, kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(up_)
    up_ = BatchNormalization(epsilon=epsilon_val)(up_)
    up_ = Dropout(drop_frac1)(up_)
    
    up_ = Convolution2D(filters, filter_size, kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(up_)
    up_ = BatchNormalization(epsilon=epsilon_val)(up_)
    
    return up_

def center_(filters,down_,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,l2_magnitude):
    
    center = Convolution2D(filters, filter_size , kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(down_)
    center = BatchNormalization(epsilon=epsilon_val)(center)
    center = Dropout(drop_frac1)(center)
    
    center = Convolution2D(filters, filter_size , kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(center)
    center = BatchNormalization(epsilon=epsilon_val)(center)

    center = Convolution2D(filters, filter_size2, kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(center)
    center = BatchNormalization(epsilon=epsilon_val)(center)

    center = Convolution2D(filters, filter_size2, kernel_regularizer=l2(l2_magnitude), padding=padding_val2, activation=activation_val1, data_format=IMAGE_ORDERING)(center)
    center = BatchNormalization(epsilon=epsilon_val)(center)
    
    return center

    
def unet2D(input_shape = IMAGE_SHAPE, LR = 0.0001, num_classes = 1):
    
    keras.backend.clear_session()
    
    inputs = Input(shape=input_shape)
    
    down1, down1_res = down(filter1, inputs,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,l2_magnitude)
    down2, down2_res = down(filter2, down1 ,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,l2_magnitude)
    down3, down3_res = down(filter3, down2 ,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,l2_magnitude)    
       
    center     =    center_(filter4, down3 ,filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,l2_magnitude)
    
    up3 =     up(filter3, center, down3_res, filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,up_scale_size,l2_magnitude)
    up2 =     up(filter2, up3   , down2_res, filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,up_scale_size,l2_magnitude)
    up1 =     up(filter1, up2   , down1_res, filter_size,filter_size2,drop_frac1,drop_frac2,stride_val,epsilon_val,padding_val2,activation_val1,up_scale_size,l2_magnitude)
    
    classify = Convolution2D(num_classes, (1, 1), activation=activation_val2, data_format=IMAGE_ORDERING)(up1)
    
    inputs_overall = [inputs]
    outputs_overall = [classify]
    
    model = Model(inputs=inputs_overall, outputs=outputs_overall)
    
    adam = Adam(lr = LEARNING_RATE)
    model.compile(loss=bce_dice_loss, optimizer=adam, metrics=[dice_coef, 'mse'])
    
    print(model.summary())
    
    return model

# with open('model_summary.txt', 'w') as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'))
#     # return model
#tf.keras.utils.plot_model(model, show_dtype=True, 
#                       show_layer_names=True, show_shapes=True,  
#                       to_file='model.png')