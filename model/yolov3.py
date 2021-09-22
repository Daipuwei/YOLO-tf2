# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午3:01
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : yolov3.py
# @Software: PyCharm


"""
    这是YOLOv3(-spp)的模型定义脚本
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
from model.layer.yolo_common import rDBL_block
from model.backbone.darknet import DarknetConv2D
from model.backbone.darknet import DarknetConv2D_BN_Leaky

def yolov3_body(feats, num_anchors, num_classes,use_spp=False):
    """
    这是YOLOv3(-spp)的主干模块定义函数
    :param feats: backbone特征张量列表
    :param num_anchors: anchor个数
    :param num_classes: 目标分类个数
    :param use_spp: 是否使用SPP模块，默认为False，若为True这则为YOLOv3-spp
    :return:
    """
    C3, C4, C5 = feats

    # P5
    P5 = rDBL_block(C5,512,use_spp=use_spp)
    P5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors*(num_classes+5), (1, 1))(P5_output)

    # P4
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                          UpSampling2D(2))(P5)
    P4 = Concatenate()([P5_upsample,C4])
    P4 = rDBL_block(P4,256,False)
    P4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors*(num_classes+5), (1, 1))(P4_output)

    # P3
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                          UpSampling2D(2))(P4)
    P3 = Concatenate()([P4_upsample,C3])
    P3 = rDBL_block(P3, 128, False)
    P3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P3)
    P3_output = DarknetConv2D(num_anchors*(num_classes+5), (1, 1))(P3_output)
    return [P5_output,P4_output,P3_output]

def build_yolov3_train(cfg):
    """
    这是搭建训练阶段YOLOv3(-spp)的函数
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
    if cfg.MODEL.BACKBONE.NAME == 'DarkNet53':               # Darknet53
        from model.backbone.darknet import yolov3_darknet53_backbone
        feats = yolov3_darknet53_backbone(image_input,base=cfg.MODEL.BACKBONE.BASE,
                                          attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)
    else:                                                   # 后续接口，首先默认为Darknet53
        from model.backbone.darknet import yolov3_darknet53_backbone
        feats = yolov3_darknet53_backbone(image_input, base=cfg.MODEL.BACKBONE.BASE,
                                          attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)

    # 搭建YOLOv3(-spp)主干
    if cfg.MODEL.USE_SPP:
        print('Create YOLOv3-spp model with {} anchors and {} classes.'.format(num_anchors,num_classes))
    else:
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    yolov3_outputs = yolov3_body(feats, num_anchors//3, num_classes,cfg.MODEL.USE_SPP)
    yolov3_body_model = Model(image_input,yolov3_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov3_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov3_body_model.summary()

    # 搭建训练阶段YOLOv3(-spp)
    h, w = cfg.INPUT.SIZE_TRAIN
    y_true = [Input(shape=(None, None, num_anchors // 3, num_classes + 5)) for l in range(3)]
    loss_input = [*yolov3_body_model.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_threshold': cfg.DATASET.IGNORE_THRESHOLD,
                                   'label_smoothing': cfg.LOSS.USE_LABEL_SMOOTHING,
                                   'use_giou_loss': cfg.LOSS.USE_GIOU_LOSS,
                                   'use_diou_loss': cfg.LOSS.USE_DIOU_LOSS,
                                   'use_ciou_loss': cfg.LOSS.USE_CIOU_LOSS,
                                   'model_name': cfg.MODEL.MODEL_NAME})(loss_input)
    yolov3_train_model = Model([image_input, *y_true], model_loss)
    return yolov3_body_model,yolov3_train_model

def build_yolov3_eval(cfg):
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

    # 搭建基础特征提取网络
    image_input = Input(shape=(None, None, 3), name='image_input')
    if cfg.MODEL.BACKBONE.NAME == 'DarkNet53':               # Darknet53
        from model.backbone.darknet import yolov3_darknet53_backbone
        feats = yolov3_darknet53_backbone(image_input,base=cfg.MODEL.BACKBONE.BASE,
                                          attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)
    else:                                                   # 后续接口，首先默认为Darknet53
        from model.backbone.darknet import yolov3_darknet53_backbone
        feats = yolov3_darknet53_backbone(image_input, base=cfg.MODEL.BACKBONE.BASE,
                                          attention_type=cfg.MODEL.BACKBONE.ATTENTION_TYPE)

    # 搭建YOLOv3(-spp)主干
    if cfg.MODEL.USE_SPP:
        print('Create YOLOv3-spp model with {} anchors and {} classes.'.format(num_anchors,num_classes))
    else:
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    yolov3_outputs = yolov3_body(feats, num_anchors//3, num_classes,cfg.MODEL.USE_SPP)
    yolov3_body_model = Model(image_input,yolov3_outputs)
    if cfg.MODEL.MODEL_PATH is not None:
        yolov3_body_model.load_weights(os.path.abspath(cfg.MODEL.MODEL_PATH), by_name=True, skip_mismatch=True)
        print('Load weights from: {}.'.format(os.path.abspath(cfg.MODEL.MODEL_PATH)))
    yolov3_body_model.summary()

    # 搭建评估阶段YOLOv3(-spp)
    input_image_shape = Input(shape=(2,), batch_size=1, name='input_image_shape')
    inputs = [*yolov3_body_model.output, input_image_shape]
    outputs = Lambda(yolo_eval, output_shape=(1,), name='yolov3_preds',
                     arguments={'anchors': anchors,
                                'num_classes': num_classes,
                                'max_boxes': cfg.DATASET.MAX_BOXES,
                                'score_threshold': cfg.DATASET.SCORE_THRESHOLD,
                                'iou_threshold': cfg.DATASET.IOU_THRESHOLD,
                                'letterbox_image': cfg.DATASET.LETTERBOX_IMAGE,
                                'model_name':cfg.MODEL.MODEL_NAME})(inputs)
    yolov3_eval_model = Model([image_input, input_image_shape], outputs)
    return yolov3_eval_model