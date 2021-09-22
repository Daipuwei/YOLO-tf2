# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:14
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : cspdarknet.py
# @Software: PyCharm

"""
    这是CSPDarkNet模型的定义脚本
"""

from functools import wraps

import tensorflow as tf
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal

from utils.model_utils import compose
from model.layer.activition import Mish
from model.backbone.darknet import DarknetConv2D_BN_Leaky

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {'kernel_initializer': RandomNormal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

def split(feature_tesnsor,group=2,group_id=1):
    '''
    这是均分特征张量的函数
    Args:
        feature_tesnsor: 特征张量
        group: 均分个数，默认为2
        group_id: 返回特征张量id，默认为1
    Returns:
    '''
    split_features = tf.split(feature_tesnsor,num_or_size_splits=group,axis=-1)
    return split_features[group_id]

def resblock_body_mish(x, num_filters, num_blocks,attention_type=None):
    '''
    这是CSPDarkNet-53中的残差模块的定义函数
    Args:
        x: 输入张量
        num_filters: 卷积层的输出通道数
        num_blocks: 模块个数
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Mish(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Mish(num_filters, (3,3))(y)
        if attention_type is None:              # 不使用注意力机制
            pass
        elif attention_type == 'se':            # SE注意力机制
            from model.layer.attention import se_attention
            y = se_attention(y)
        else:                                   # 后续接口
            pass
        x = Add()([x,y])
    return x

def csp_resblock_body_mish(x, num_filters, num_blocks, all_narrow=True,attention_type=None):
    '''
     这是CSDDarknet-53中CSP残差模块的定义函数
    Args:
        x: 输入张量
        num_filters: 卷积层的输出通道数
        num_blocks: 模块个数
        all_narrow: 是否使用窄通道标志位
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''

    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(preconv1)

    shortconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    mainconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(mainconv)
        if attention_type is None:                              # 不使用注意力机制
            pass
        elif attention_type == 'se':                            # SE注意力机制
            from model.layer.attention import se_attention
            y = se_attention(y)
        else:                                                   # 后续接口
            pass
        mainconv = Add()([mainconv, y])
    postconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(mainconv)
    route = Concatenate()([postconv, shortconv])
    return DarknetConv2D_BN_Mish(num_filters, (1, 1))(route)

def tiny_resblock_body_leaky(x, num_filters,attention_type):
    '''
    这是yolov4-tiny中残差模块的初始化函数
    Args:
        x: 输入张量
        num_filters: 卷积层的输出通道数
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    conv1 = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(x)
    split_feature1 = Lambda(split,arguments={'group':2,
                                             'group_id':1})(conv1)
    x = DarknetConv2D_BN_Leaky(num_filters//2,(3,3))(split_feature1)
    split_feature2 = x
    x = DarknetConv2D_BN_Leaky(num_filters//2,(3,3))(x)
    x = Concatenate()([split_feature2,x])
    x = DarknetConv2D_BN_Leaky(num_filters,(1,1))(x)
    if attention_type is None:  # 不使用注意力机制
        pass
    elif attention_type == 'se':  # SE注意力机制
        from model.layer.attention import se_attention
        x = se_attention(x)
    else:  # 后续接口
        pass
    feat = x
    x = Concatenate()([conv1,x])
    x = MaxPooling2D(pool_size=(2,2))(x)
    return x,feat

def yolov4_cspdarknet_backbone(image_input,base=8,attention_type=None):
    '''
    这是YOLOv4中CSPDarknet-53的backbone定义函数
    Args:
        image_input: 图像输入
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Mish(base*4, (3, 3))(image_input)
    x = csp_resblock_body_mish(x, base*8, 1, False, attention_type)
    x = csp_resblock_body_mish(x, base*16, 2, True, attention_type)
    x = csp_resblock_body_mish(x, base*32, 8, True, attention_type)
    feat1 = x
    x = csp_resblock_body_mish(x, base*64, 8, True, attention_type)
    feat2 = x
    x = csp_resblock_body_mish(x, base*128, 4, True, attention_type)
    feat3 = x
    return feat1, feat2, feat3

def yolov4_csp_cspdarknet_backbone(image_input,base=8,attention_type=None):
    '''
     这是YOLOv4-csp中CSPDarknet-53的backbone定义函数
    Args:
        image_input: 图像输入
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Mish(base*4, (3, 3))(image_input)
    x = resblock_body_mish(x, base*8, 1, attention_type)
    x = csp_resblock_body_mish(x, base*16, 2, True, attention_type)
    x = csp_resblock_body_mish(x, base*32, 8, True, attention_type)
    feat1 = x
    x = csp_resblock_body_mish(x, base*64, 8, True, attention_type)
    feat2 = x
    x = csp_resblock_body_mish(x, base*128, 4, True, attention_type)
    feat3 = x
    return feat1,feat2,feat3

def yolov4_p5_cspdarknet_backbone(image_input,base=8,attention_type=None):
    '''
    这是YOLOv4-p5中CSPDarknet的backbone定义函数
    Args:
        image_input: 图像输入
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Mish(base*4, (3, 3))(image_input)
    x = csp_resblock_body_mish(x, base*8, 1, True, attention_type)
    x = csp_resblock_body_mish(x, base*16, 3, True, attention_type)
    x = csp_resblock_body_mish(x, base*32, 15, True, attention_type)
    feat1 = x
    x = csp_resblock_body_mish(x, base*64, 15, True, attention_type)
    feat2 = x
    x = csp_resblock_body_mish(x, base*128, 7, True, attention_type)
    feat3 = x
    return feat1,feat2,feat3

def yolov4_p6_cspdarknet_backbone(image_input,base=8,attention_type=None):
    '''
    这是YOLOv4-p6中CSPDarknet的backbone定义函数
    Args:
        image_input: 图像输入
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Mish(base*4, (3, 3))(image_input)
    x = csp_resblock_body_mish(x, base*8, 1, True, attention_type)
    x = csp_resblock_body_mish(x, base*16, 3, True, attention_type)
    x = csp_resblock_body_mish(x, base*32, 15, True, attention_type)
    feat1 = x
    x = csp_resblock_body_mish(x, base*64, 15, True, attention_type)
    feat2 = x
    x = csp_resblock_body_mish(x, base*128, 7, True, attention_type)
    feat3 = x
    x = csp_resblock_body_mish(x, base*128, 7, True, attention_type)
    feat4 = x
    return feat1,feat2,feat3,feat4

def yolov4_p7_cspdarknet_backbone(image_input,base=8,attention_type=None):
    '''
    这是YOLOv4-p7中CSPDarknet的backbone定义函数
    Args:
        image_input: 图像输入
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Mish(base*4, (3, 3))(image_input)
    x = csp_resblock_body_mish(x, base*8, 1, True, attention_type)
    x = csp_resblock_body_mish(x, base*16, 3, True, attention_type)
    x = csp_resblock_body_mish(x, base*32, 15, True, attention_type)
    feat1 = x
    x = csp_resblock_body_mish(x, base*64, 15, True, attention_type)
    feat2 = x
    x = csp_resblock_body_mish(x, base*128, 7, True, attention_type)
    feat3 = x
    x = csp_resblock_body_mish(x, base*128, 7, True, attention_type)
    feat4 = x
    x = csp_resblock_body_mish(x, base*128, 7, True, attention_type)
    feat5 = x
    return feat1,feat2,feat3,feat4,feat5

def yolov4_tiny_cspdarknet_backbone(image_input,base=8,attention_type=None):
    '''
    这是YOLOv4-tiny中CSPDarknet的backbone定义函数
    Args:
        image_input: 图像输入
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(image_input)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3, 3), strides=(2, 2))(x)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2))(x)

    x, _ = tiny_resblock_body_leaky(x, base*8)
    x, _ = tiny_resblock_body_leaky(x, base*16)
    x, feat1 = tiny_resblock_body_leaky(x, base*32,attention_type)
    x = DarknetConv2D_BN_Leaky(base*64, (3, 3))(x)
    feat2 = x
    return feat1, feat2