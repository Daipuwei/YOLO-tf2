# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:15
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : activition.py
# @Software: PyCharm

"""
    这是激活函数层的定义脚本
"""

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Mish(Layer):

    '''
        这是Mish激活函数层的类定义
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape