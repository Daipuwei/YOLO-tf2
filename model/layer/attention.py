# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:21
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : attention.py
# @Software: PyCharm

'''
    这是各种注意力机制的定义脚本
'''

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras import backend as K

def se_attention(tensor, ratio=16):
    '''
    这是SE注意力机制模块的定义函数
    Args:
        tensor: 输入张量
        ratio: 下采样率，默认为16
    Returns:
    '''
    init = tensor
    filters = K.int_shape(init)[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = LeakyReLU(alpha=0.1)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])
    return x