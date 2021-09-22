# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:21
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : darknet.py
# @Software: PyCharm


"""
    这是DarkNet模型的定义脚本
"""

from functools import wraps
from utils.model_utils import compose

from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

@wraps(Conv2D)          # 利用DarknetConv2D来装饰Conv2D
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs['kernel_initializer'] = 'he_normal'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body_leaky(x, num_filters, num_blocks,attention_type=None):
    '''
    这是DarkNet-53中的残差模块的定义函数
    Args:
        x: 输入张量
        num_filters: 卷积层的输出通道数
        num_blocks: 模块个数
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        if attention_type is None:              # 不使用注意力机制
            pass
        elif attention_type == 'se':            # SE注意力机制
            from model.layer.attention import se_attention
            y = se_attention(y)
        else:                                   # 后续接口
            pass
        x = Add()([x,y])
    return x

def yolov3_darknet53_backbone(image_input,base=8,attention_type=None):
    '''
    这是YOLOv3的backbone函数
    Args:
        image_input: 图像输入张量
        base: 卷积基数，默认为8，可以控制卷积层通道数大小，在一定程度上可以实现模型剪枝
        attention_type: 注意力机制类型，默认为None，即不使用注意力机制
    Returns:
    '''
    x = DarknetConv2D_BN_Leaky(base*4, (3,3))(image_input)
    x = resblock_body_leaky(x, base*8, 1, attention_type)
    x = resblock_body_leaky(x, base*16, 2, attention_type)
    x = resblock_body_leaky(x, base*32, 8, attention_type)
    feat1 = x
    x = resblock_body_leaky(x, base*64, 8,attention_type)
    feat2 = x
    x = resblock_body_leaky(x, base*128, 4, attention_type)
    feat3 = x
    return feat1, feat2, feat3
