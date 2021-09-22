# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午10:41
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : defualts.py
# @Software: PyCharm


"""
    这是默认参数字典的定义脚本
"""

from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = "yolov3"
# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.MODEL_PATH = None
_C.MODEL.USE_SPP = False
# backbone
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'DarkNet53'
_C.MODEL.BACKBONE.BASE = 8
_C.MODEL.BACKBONE.ATTENTION_TYPE = None

# loss
_C.LOSS = CN()
_C.LOSS.USE_LABEL_SMOOTHING = True
_C.LOSS.USE_GIOU_LOSS = False
_C.LOSS.USE_DIOU_LOSS = False
_C.LOSS.USE_CIOU_LOSS = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [608,608]
# Size of the image during test
_C.INPUT.SIZE_VAL = [608,608]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET_NAME=""
_C.DATASET.TRAIN_TXT_PATH = ''
_C.DATASET.VAL_TXT_PATH = ''
_C.DATASET.ANCHORS_PATH = ''
_C.DATASET.CLASSES_PATH = ''
_C.DATASET.MAX_BOXES = 20
_C.DATASET.IOU_THRESHOLD = 0.5
_C.DATASET.SCORE_THRESHOLD = 0.6
_C.DATASET.IGNORE_THRESHOLD = 0.5
_C.DATASET.LETTERBOX_IMAGE = True
_C.DATASET.USE_MOSAIC = False
_C.DATASET.DROP_LAST_BATCH = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Optimizer
_C.SOLVER.OPITIMIZER_NAME = "Adam"
_C.SOLVER.EPOCH = 500
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.LEARNING_RATE = 1e-3

# ReduceLROnPlateau learning rate options
_C.SOLVER.SCHED = CN()
_C.SOLVER.SCHED.NAME = 'ReduceLROnPlateau'
_C.SOLVER.SCHED.FACTOR = 0.1
_C.SOLVER.SCHED.PATIENCE = 2

# EarlyStopping options
_C.SOLVER.EARLYSTOPPING = CN()
_C.SOLVER.EARLYSTOPPING.MIN_DELTA = 1e-6
_C.SOLVER.EARLYSTOPPING.PATIENCE = 20

_C.LOGS_DIR = "logs/"
_C.CHECKPOINTS_DIR = "checkpoints/"
_C.FONT_PATH = "model_data/simhei.ttf"

def get_cfg_defaults():
    """
    这是复制一份
    :return:
    """
    return _C.clone()