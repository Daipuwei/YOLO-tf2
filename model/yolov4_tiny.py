# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午5:43
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : yolov4_tiny.py
# @Software: PyCharm

"""
    这是YOLOv4-tiny模型定义脚本
"""

import os
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D

from utils.model_utils import compose
from utils.model_utils import get_anchors
from utils.model_utils import get_classes

from model.loss import yolo_loss
from model.layer.yolo_common import yolo_eval
from model.backbone.darknet import DarknetConv2D
from model.backbone.darknet import DarknetConv2D_BN_Leaky

def yolov4_tiny_body(feats, num_anchors, num_classes):
    """
    这是YOLOv4-tiny的主干PANet定义函数
    :param feats: backbone输入特征张量列表
    :param num_anchors: anchor个数
    :param num_classes: 目标个数
    :return:
    """
    C4,C5 = feats

    # P5
    P5 = DarknetConv2D_BN_Leaky(256, (1, 1))(C5)
    P5_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)

    # P5
    P5_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(P5)
    P4 = Concatenate()([P5_upsample, C4])
    P4_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)

    return [P5_output, P4_output]

def build_yolov4_tiny_train(cfg):
    """
    这是搭建训练阶段YOLOv4-tiny的函数
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
        from model.backbone.cspdarknet import yolov4_tiny_cspdarknet_backbone
        feats = yolov4_tiny_cspdarknet_backbone(image_input,base=cfg.MODEL.BACKBONE.BASE)
    else:                                                       # 后续接口，首先默认为CSPDarknet
        from model.backbone.cspdarknet import yolov4_tiny_cspdarknet_backbone
        feats = yolov4_tiny_cspdarknet_backbone(image_input, base=cfg.MODEL.BACKBONE.BASE)

    # 搭建YOLOv4主干
    print('Create YOLOv4-tiny model with {} anchors and {} classes.'.format(num_anchors,num_classes))
    yolov4_tiny_outputs = yolov4_tiny_body(feats, num_anchors//2, num_classes)
    yolov4_tiny_body_model = Model(image_input,yolov4_tiny_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov4_tiny_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov4_tiny_body_model.summary()

    # 搭建训练阶段YOLOv4
    y_true = [Input(shape=(None,None,num_anchors//2, num_classes+5)) for l in range(2)]
    loss_input = [*yolov4_tiny_body_model.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_threshold': cfg.DATASET.IGNORE_THRESHOLD,
                                   'label_smoothing': cfg.LOSS.USE_LABEL_SMOOTHING,
                                   'use_giou_loss': cfg.LOSS.USE_GIOU_LOSS,
                                   'use_diou_loss': cfg.LOSS.USE_DIOU_LOSS,
                                   'use_ciou_loss': cfg.LOSS.USE_CIOU_LOSS,
                                   'model_name': cfg.MODEL.MODEL_NAME})(loss_input)
    yolov4_tiny_train_model = Model([image_input, *y_true], model_loss)
    return yolov4_tiny_body_model,yolov4_tiny_train_model

def build_yolov4_eval(cfg):
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
    if cfg.MODEL.BACKBONE.NAME == 'CPSDarkNet':               # CSPDarknet
        from model.backbone.cspdarknet import yolov4_tiny_cspdarknet_backbone
        feats = yolov4_tiny_cspdarknet_backbone(image_input,base=cfg.MODEL.BACKBONE.BASE)
    else:                                                       # 后续接口，首先默认为CSPDarknet
        from model.backbone.cspdarknet import yolov4_tiny_cspdarknet_backbone
        feats = yolov4_tiny_cspdarknet_backbone(image_input, base=cfg.MODEL.BACKBONE.BASE)

    # 搭建YOLOv4主干
    print('Create YOLOv4-tiny model with {} anchors and {} classes.'.format(num_anchors,num_classes))
    yolov4_tiny_outputs = yolov4_tiny_body(feats, num_anchors//2, num_classes)
    yolov4_tiny_body_model = Model(image_input,yolov4_tiny_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov4_tiny_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov4_tiny_body_model.summary()

    # 搭建评估阶段YOLOv4
    input_image_shape = Input(shape=(2,), batch_size=1, name='input_image_shape')
    inputs = [*yolov4_tiny_body_model.output, input_image_shape]
    outputs = Lambda(yolo_eval, output_shape=(1,), name='yolov4_preds',
                     arguments={'anchors': anchors,
                                'num_classes': num_classes,
                                'max_boxes': cfg.DATASET.MAX_BOXES,
                                'score_threshold': cfg.DATASET.SCORE_THRESHOLD,
                                'iou_threshold': cfg.DATASET.IOU_THRESHOLD,
                                'letterbox_image': cfg.DATASET.LETTERBOX_IMAGE,
                                'model_name':cfg.MODEL.MODEL_NAME})(inputs)
    yolov4_tiny_eval_model = Model([image_input, input_image_shape], outputs)
    return yolov4_tiny_eval_model