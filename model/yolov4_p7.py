# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午4:42
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : yolov4_p7.py
# @Software: PyCharm

"""
    这是YOLOv4-p7模型的定义脚本
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D

from utils.model_utils import compose
from utils.model_utils import get_anchors
from utils.model_utils import get_classes

from model.loss import yolo_loss
from model.layer.yolo_common import yolo_eval
from model.layer.yolo_common import rCSP_block
from model.backbone.cspdarknet import DarknetConv2D
from model.backbone.cspdarknet import DarknetConv2D_BN_Mish

def yolov4_p7_body(feats, num_anchors, num_classes):
    """
    这是YOLOv4-p7主干的定义函数
    :param feats: 输入特征张量列表
    :param num_anchors: anchor个数
    :param num_classes: 目标分类个数
    :return:
    """
    C3, C4, C5, C6, C7 = feats

    # P7
    P7 = rCSP_block(C7, 512, True)

    # P6
    P7_upsample = compose(DarknetConv2D_BN_Mish(512, (1, 1)),
                          UpSampling2D(2))(P7)
    P6 = DarknetConv2D_BN_Mish(512, (1, 1))(C6)
    P6 = Concatenate()([P6, P7_upsample])
    P6 = rCSP_block(P6, 512, False)

    # P5
    P6_upsample = compose(DarknetConv2D_BN_Mish(512, (1, 1)),
                          UpSampling2D(2))(P6)
    P5 = DarknetConv2D_BN_Mish(512, (1, 1))(C5)
    P5 = Concatenate()([P5, P6_upsample])
    P5 = rCSP_block(P5, 512, False)

    # P4
    P5_upsample = compose(DarknetConv2D_BN_Mish(256, (1, 1)),
                          UpSampling2D(2))(P5)
    P4 = DarknetConv2D_BN_Mish(256, (1, 1))(C4)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = rCSP_block(P4, 256, False)

    # P3
    P4_upsample = compose(DarknetConv2D_BN_Mish(128, (1, 1)),
                          UpSampling2D(2))(P4)
    P3 = DarknetConv2D_BN_Mish(128, (1, 1))(C3)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = rCSP_block(P3, 128, False)
    P3_output = DarknetConv2D_BN_Mish(256, (3, 3))(P3)
    P3_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P3_output)

    # P4
    P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
    P3_downsample = DarknetConv2D_BN_Mish(256, (3, 3), strides=(2, 2))(P3_downsample)
    P4 = Concatenate()([P3_downsample, P4])
    P4 = rCSP_block(P4, 256, False)
    P4_output = DarknetConv2D_BN_Mish(512, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P4_output)

    # P5
    P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
    P4_downsample = DarknetConv2D_BN_Mish(512, (3, 3), strides=(2, 2))(P4_downsample)
    P5 = Concatenate()([P4_downsample, P5])
    P5 = rCSP_block(P5, 512, False)
    P5_output = DarknetConv2D_BN_Mish(1024, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P5_output)

    # P6
    P5_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P5)
    P5_downsample = DarknetConv2D_BN_Mish(512, (3, 3), strides=(2, 2))(P5_downsample)
    P6 = Concatenate()([P5_downsample, P6])
    P6 = rCSP_block(P6, 512, False)
    P6_output = DarknetConv2D_BN_Mish(1024, (3, 3))(P6)
    P6_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P6_output)

    P6_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P6)
    P6_downsample = DarknetConv2D_BN_Mish(512, (3, 3), strides=(2, 2))(P6_downsample)
    P7 = Concatenate()([P6_downsample, P7])
    P7 = rCSP_block(P7, 512, False)
    P7_output = DarknetConv2D_BN_Mish(1024, (3, 3))(P7)
    P7_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1),
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P7_output)

    return [P7_output,P6_output,P5_output, P4_output, P3_output]

def build_yolov4_p7_train(cfg):
    """
    这是搭建训练阶段YOLO v4的函数
    :param cfg: 参数配置类
    :return:
    """
    # 初始化anchor和classes
    anchors = get_anchors(cfg.DATASET.ANCHORS_PATH)
    classes = get_classes(cfg.DATASET.CLASSES_PATH)
    num_anchors = len(anchors)
    num_classes = len(classes)

    # 搭建基础特征提取网络
    image_input = Input(shape=(None, None, 3), name='image_input')
    if cfg.MODEL.BACKBONE.NAME == 'CPSDarkNet':               # CSPDarknet
        from model.backbone.cspdarknet import yolov4_p7_cspdarknet_backbone
        feats = yolov4_p7_cspdarknet_backbone(image_input,base=cfg.MODEL.BACKBONE.BASE,
                                              attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)
    else:                                                       # 后续接口，首先默认为CSPDarknet53
        from model.backbone.cspdarknet import yolov4_p7_cspdarknet_backbone
        feats = yolov4_p7_cspdarknet_backbone(image_input, base=cfg.MODEL.BACKBONE.BASE,
                                              attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)

    # 搭建YOLOv4-p7主干
    print('Create YOLOv4-p7 model with {} anchors and {} classes.'.format(num_anchors,num_classes))
    yolov4_p7_outputs = yolov4_p7_body(feats, num_anchors//5, num_classes)
    yolov4_p7_body_model = Model(image_input,yolov4_p7_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov4_p7_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov4_p7_body_model.summary()

    # 搭建训练阶段YOLOv4-p7
    y_true = [Input(shape=(None, None, num_anchors//5, num_classes+5)) for l in range(5)]
    loss_input = [*yolov4_p7_body_model.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_threshold': cfg.DATASET.IGNORE_THRESHOLD,
                                   'label_smoothing': cfg.LOSS.USE_LABEL_SMOOTHING,
                                   'use_giou_loss': cfg.LOSS.USE_GIOU_LOSS,
                                   'use_diou_loss': cfg.LOSS.USE_DIOU_LOSS,
                                   'use_ciou_loss': cfg.LOSS.USE_CIOU_LOSS,
                                   'model_name': cfg.MODEL.MODEL_NAME})(loss_input)
    yolov4_p7_train_model = Model([image_input, *y_true], model_loss)
    return yolov4_p7_body_model,yolov4_p7_train_model

def build_yolov4_p7_eval(cfg):
    """
    这是搭建评估阶段YOLO v4的函数
    :param cfg: 参数配置类
    :return:
    """
    # 初始化anchor和classes
    anchors = get_anchors(cfg.DATASET.ANCHORS_PATH)
    classes = get_classes(cfg.DATASET.CLASSES_PATH)
    num_anchors = len(anchors)
    num_classes = len(classes)

    # 搭建基础特征提取网络
    image_input = Input(shape=(None, None, 3), name='image_input')
    if cfg.MODEL.BACKBONE.NAME == 'CPSDarkNet':               # CSPDarknet53
        from model.backbone.cspdarknet import yolov4_p7_cspdarknet_backbone
        feats = yolov4_p7_cspdarknet_backbone(image_input,base=cfg.MODEL.BACKBONE.BASE,
                                              attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)
    else:                                                       # 后续接口，首先默认为CSPDarknet53
        from model.backbone.cspdarknet import yolov4_p7_cspdarknet_backbone
        feats = yolov4_p7_cspdarknet_backbone(image_input, base=cfg.MODEL.BACKBONE.BASE,
                                              attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)

    # 搭建YOLOv4-p7主干
    print('Create YOLOv4-p7 model with {} anchors and {} classes.'.format(num_anchors,num_classes))
    yolov4_p7_outputs = yolov4_p7_body(feats, num_anchors//5, num_classes)
    yolov4_p7_body_model = Model(image_input,yolov4_p7_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov4_p7_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov4_p7_body_model.summary()

    # 搭建评估阶段YOLOv4-p7
    input_image_shape = Input(shape=(2,), batch_size=1, name='input_image_shape')
    inputs = [*yolov4_p7_body_model.output, input_image_shape]
    outputs = Lambda(yolo_eval, output_shape=(1,), name='yolov4_p7_preds',
                     arguments={'anchors': anchors,
                                'num_classes': num_classes,
                                'max_boxes': cfg.DATASET.MAX_BOXES,
                                'score_threshold': cfg.DATASET.SCORE_THRESHOLD,
                                'iou_threshold': cfg.DATASET.IOU_THRESHOLD,
                                'letterbox_image': cfg.DATASET.LETTERBOX_IMAGE,
                                'model_name':cfg.MODEL.MODEL_NAME})(inputs)
    yolov4_p7_eval_model = Model([image_input, input_image_shape], outputs)
    return yolov4_p7_eval_model