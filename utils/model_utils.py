# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:18
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : model_utils.py
# @Software: PyCharm


"""
    这是YOLO模型训练过程相关工具函数与工具类的定义脚本
"""

import json
import numpy as np
from functools import reduce

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def get_classes(classes_path):
    '''
    这是获取目标分类名称的函数
    Args:
        classes_path: 目标分类名称txt文件路径
    Returns:
    '''
    classes_names = []
    with open(classes_path, 'r') as f:
        for line in f.readlines():
            classes_names.append(line.strip())
    classes_names = np.array(classes_names)
    return classes_names

def get_anchors(anchors_path):
    '''
    这是获取anchor尺寸的函数
    Args:
        anchors_path: anchor尺寸txt文件路径
    Returns:
    '''
    with open(anchors_path, 'r') as f:
        line = f.readline()
        anchors = [float(x) for x in line.split(',')]
    anchors = np.array(anchors)
    anchors = np.reshape(anchors, (-1, 2))
    return anchors

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)