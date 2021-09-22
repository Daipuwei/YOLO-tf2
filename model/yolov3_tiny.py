# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午5:43
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : yolov3_tiny.py
# @Software: PyCharm


"""
    这是YOLOv3-tiny模型的定义脚本
"""

import os
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D

from utils.model_utils import compose
from utils.model_utils import get_anchors
from utils.model_utils import get_classes

from model.loss import yolo_loss
from model.layer.yolo_common import yolo_eval
from model.backbone.darknet import DarknetConv2D
from model.backbone.darknet import DarknetConv2D_BN_Leaky

def yolov3_tiny_body(image_input, num_anchors, num_classes):
    """
    这是YOLOv3-tiny的主干模块定义函数
    :param image_input: 输入张量
    :param num_anchors: anchor个数
    :param num_classes: 目标分类个数
    :return:
    """
    x1 = compose(DarknetConv2D_BN_Leaky(16, (3,3)),
                 MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                 DarknetConv2D_BN_Leaky(32, (3,3)),
                 MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                 DarknetConv2D_BN_Leaky(64, (3,3)),
                 MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                 DarknetConv2D_BN_Leaky(128, (3,3)),
                 MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                 DarknetConv2D_BN_Leaky(256, (3,3)))(image_input)
    x2 = compose(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                 DarknetConv2D_BN_Leaky(512, (3,3)),
                 MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
                 DarknetConv2D_BN_Leaky(1024, (3,3)),
                 DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(DarknetConv2D_BN_Leaky(512, (3,3)),
                 DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)
    x2 = compose(DarknetConv2D_BN_Leaky(128, (1,1)),
                 UpSampling2D(2))(x2)
    y2 = compose(Concatenate(),
                 DarknetConv2D_BN_Leaky(256, (3,3)),
                 DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])
    print()
    return [y1,y2]

def build_yolov3_tiny_train(cfg):
    """
    这是搭建训练阶段YOLOv3-tiny的函数
    :param cfg: 参数配置类
    :return:
    """
    # 初始化anchor和classes
    anchors = get_anchors(cfg.DATASET.ANCHORS_PATH)
    classes = get_classes(cfg.DATASET.CLASSES_PATH)
    num_anchors = len(anchors)
    num_classes = len(classes)

    # 搭建YOLOv3-tiny主干
    image_input = Input(shape=(None, None, 3), name='image_input')
    print('Create YOLOv3-tiny model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    yolov3_tiny_outputs = yolov3_tiny_body(image_input, num_anchors//2, num_classes)
    yolov3_tiny_body_model = Model(image_input, yolov3_tiny_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov3_tiny_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov3_tiny_body_model.summary()

    # 搭建训练阶段YOLOv3-tiny
    y_true = [Input(shape=(None, None, num_anchors//2, num_classes + 5)) for l in range(2)]
    loss_input = [*yolov3_tiny_body_model.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'label_smoothing': cfg.LOSS.USE_LABEL_SMOOTHING,
                                   'use_giou_loss': cfg.LOSS.USE_GIOU_LOSS,
                                   'use_diou_loss': cfg.LOSS.USE_DIOU_LOSS,
                                   'use_ciou_loss': cfg.LOSS.USE_CIOU_LOSS,
                                   'model_name': cfg.MODEL.MODEL_NAME})(loss_input)
    yolov3_tiny_train_model = Model([image_input, *y_true], model_loss)
    return yolov3_tiny_body_model, yolov3_tiny_train_model

def build_yolov3_tiny_eval(cfg):
    """
    这是搭建评估阶段YOLOv3(-spp)的函数
    :param cfg: 参数配置类
    :return:
    """
    # 初始化anchor和classes
    anchors = get_anchors(cfg.DATASET.ANCHORS_PATH)
    classes = get_classes(cfg.DATASET.CLASSES_PATH)
    num_anchors = len(anchors)
    num_classes = len(classes)

    # 搭建YOLOv3-tiny主干
    image_input = Input(shape=(None, None, 3), name='image_input')
    print('Create YOLOv3-tiny model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    yolov3_tiny_outputs = yolov3_tiny_body(image_input, num_anchors//2, num_classes)
    yolov3_tiny_body_model = Model(image_input, yolov3_tiny_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov3_tiny_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov3_tiny_body_model.summary()

    # 搭建评估阶段YOLOv3-tiny
    input_image_shape = Input(shape=(2,), batch_size=1, name='input_image_shape')
    inputs = [*yolov3_tiny_body_model.output, input_image_shape]
    outputs = Lambda(yolo_eval, output_shape=(1,), name='yolov3_tiny_preds',
                     arguments={'anchors': anchors,
                                'num_classes': num_classes,
                                'max_boxes': cfg.DATASET.MAX_BOXES,
                                'score_threshold': cfg.DATASET.SCORE_THRESHOLD,
                                'iou_threshold': cfg.DATASET.IOU_THRESHOLD,
                                'letterbox_image': cfg.DATASET.LETTERBOX_IMAGE,
                                'model_name': cfg.MODEL.MODEL_NAME})(inputs)
    yolov3_tiny_eval_model = Model([image_input, input_image_shape], outputs)
    return yolov3_tiny_eval_model