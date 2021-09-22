# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:26
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : train_utils.py
# @Software: PyCharm

"""
    这是不同YOLO模型训练类定义脚本
"""

import os
import yaml
import datetime

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.dataset_utils import Dataset
from utils.model_utils import get_classes
from utils.model_utils import get_anchors

class Trainer(object):

    def __init__(self, cfg):
        '''
        这是模型训练类的初始化函数
        Args:
            cfg: 参数字典
        '''
        self.cfg = cfg

        # 初始化anchor和classes
        self.anchors = get_anchors(os.path.abspath(self.cfg.DATASET.ANCHORS_PATH))
        self.classes_names = get_classes(os.path.abspath(self.cfg.DATASET.CLASSES_PATH))
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.classes_names)

        # 搭建模型
        if self.cfg.MODEL.MODEL_NAME == 'yolov3':               # 搭建YOLOv3(-spp)模型
            from model.yolov3 import build_yolov3_train
            self.body_model,self.train_model = build_yolov3_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov3-tiny':  # 搭建YOLOv4模型
            from model.yolov3_tiny import build_yolov3_tiny_train
            self.body_model, self.train_model = build_yolov3_tiny_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4':             # 搭建YOLOv4模型
            from model.yolov4 import build_yolov4_train
            self.body_model,self.train_model = build_yolov4_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-tiny':        # 搭建YOLOv4-tiny模型
            from model.yolov4_tiny import build_yolov4_tiny_train
            self.body_model,self.train_model = build_yolov4_tiny_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-csp':         # 搭建YOLOv4-csp模型
            from model.yolov4_csp import build_yolov4_csp_train
            self.body_model,self.train_model = build_yolov4_csp_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-p5':          # 搭建YOLOv4-p5模型
            from model.yolov4_p5 import build_yolov4_p5_train
            self.body_model,self.train_model = build_yolov4_p5_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-p6':          # 搭建YOLOv4-p6模型
            from model.yolov4_p6 import build_yolov4_p6_train
            self.body_model, self.train_model = build_yolov4_p6_train(self.cfg)
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-p7':          # 搭建YOLOv4-p7模型
            from model.yolov4_p7 import build_yolov4_p7_train
            self.body_model, self.train_model = build_yolov4_p7_train(self.cfg)
        else:                                                   # 预留接口，默认为搭建YOLOv3模型
            from model.yolov3 import build_yolov3_train
            self.body_model,self.train_model = build_yolov3_train(self.cfg)

    def train(self):
        '''
        这是训练模型的函数
        Returns:
        '''
        # 初始化训练、验证数据集生成器
        train_dataset = Dataset(self.cfg.DATASET.TRAIN_TXT_PATH, self.cfg.DATASET.CLASSES_PATH,
                                self.cfg.DATASET.ANCHORS_PATH, self.cfg.SOLVER.BATCH_SIZE,
                                self.cfg.INPUT.SIZE_TRAIN, self.cfg.DATASET.MAX_BOXES,
                                random=True,use_mosaic=self.cfg.DATASET.USE_MOSAIC,model_name=self.cfg.MODEL.MODEL_NAME)
        train_iter_num = train_dataset.iter_num
        train_datagen = train_dataset.generator()
        val_dataset = Dataset(self.cfg.DATASET.VAL_TXT_PATH, self.cfg.DATASET.CLASSES_PATH,
                              self.cfg.DATASET.ANCHORS_PATH, self.cfg.SOLVER.BATCH_SIZE,
                              self.cfg.INPUT.SIZE_VAL, self.cfg.DATASET.MAX_BOXES,
                              random=True,use_mosaic=False,model_name=self.cfg.MODEL.MODEL_NAME)
        val_iter_num = val_dataset.iter_num
        val_datagen = val_dataset.generator()

        # 初始化相关文件目录路径
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(self.cfg.CHECKPOINTS_DIR, time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_dir = os.path.join(self.cfg.LOGS_DIR, time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 保存配置文件
        with open(os.path.join(log_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(self.cfg, f)

        # 定义回调函数
        tensorboard = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir,
                                                  'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                     monitor='val_loss', save_weights_only=True,
                                     save_best_only=False, period=1, verbose=1)
        early_stopping = EarlyStopping(min_delta=self.cfg.SOLVER.EARLYSTOPPING.MIN_DELTA,
                                            patience=self.cfg.SOLVER.EARLYSTOPPING.PATIENCE)
        if self.cfg.SOLVER.SCHED.NAME == 'ReduceLROnPlateau':               # ReduceLROnPlateau
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            lr_scheduler = ReduceLROnPlateau(factor=self.cfg.SOLVER.SCHED.FACTOR,
                                             patience=self.cfg.SOLVER.SCHED.PATIENCE, verbose=1, min_delta=1e-8)
        else:
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            lr_scheduler = ReduceLROnPlateau(factor=0.1,patience=2, verbose=1, min_delta=1e-8)

        # 开始模型训练
        print('\n----------- start to train -----------\n')
        if True:
            if self.cfg.SOLVER.OPITIMIZER_NAME == 'Adam':
                from tensorflow.keras.optimizers import Adam
                self.optimizer = Adam(learning_rate=self.cfg.SOLVER.LEARNING_RATE)
            elif self.cfg.SOLVER.OPITIMIZER_NAME == 'Adadelta':
                from tensorflow.keras.optimizers import Adadelta
                self.optimizer = Adadelta(learning_rate=self.cfg.SOLVER.LEARNING_RATE)
            else:
                from tensorflow.keras.optimizers import Adam
                self.optimizer = Adam(learning_rate=1e-3)
            self.train_model.compile(optimizer=self.optimizer,
                                     loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            initial_epoch = 0
            history1 = self.train_model.fit_generator(generator=train_datagen,
                                                      steps_per_epoch=train_iter_num,
                                                      validation_data=val_datagen,
                                                      validation_steps=val_iter_num,
                                                      epochs=self.cfg.SOLVER.EPOCH,
                                                      initial_epoch=initial_epoch,
                                                      callbacks=[tensorboard, checkpoint,
                                                                 lr_scheduler, early_stopping])
            self.train_model.save_weights(os.path.join(checkpoint_dir, 'trained_yolo_model_stage_1.h5'))

        # 固定YOLO模型部分参数
        if self.cfg.MODEL.MODEL_NAME == 'yolov3':               # 搭建YOLOv3(-spp)模型
            size = len(self.body_model.layers) - 3
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4':             # 搭建YOLOv4模型
            size = len(self.body_model.layers) - 3
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-tiny':        # 搭建YOLOv4-tiny模型
            size = len(self.body_model.layers) - 2
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-csp':         # 搭建YOLOv4-csp模型
            size = len(self.body_model.layers) - 3
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-p5':          # 搭建YOLOv4-p5模型
            size = len(self.body_model.layers) - 3
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-p6':          # 搭建YOLOv4-p6模型
            size = len(self.body_model.layers) - 4
        elif self.cfg.MODEL.MODEL_NAME == 'yolov4-p7':          # 搭建YOLOv4-p7模型
            size = len(self.body_model.layers) - 5
        else:                                                   # 预留接口，默认为搭建YOLOv3模型
            size = len(self.body_model.layers) - 3
        for i in range(size):
            self.train_model.layers[i].trainable = False

        # 解冻后训练
        if True:
            if self.cfg.SOLVER.OPITIMIZER_NAME == 'Adam':
                from tensorflow.keras.optimizers import Adam
                self.optimizer = Adam(learning_rate=self.cfg.SOLVER.LEARNING_RATE/10)
            elif self.cfg.SOLVER.OPITIMIZER_NAME == 'Adadelta':
                from tensorflow.keras.optimizers import Adadelta
                self.optimizer = Adadelta(learning_rate=self.cfg.SOLVER.LEARNING_RATE/10)
            else:
                from tensorflow.keras.optimizers import Adam
                self.optimizer = Adam(learning_rate=1e-4)
            self.train_model.compile(optimizer=self.optimizer,
                                     loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            initial_epoch = len(history1.history['loss'])
            history2 = self.train_model.fit_generator(generator=train_datagen,
                                                      steps_per_epoch=train_iter_num,
                                                      validation_data=val_datagen,
                                                      validation_steps=val_iter_num,
                                                      epochs=self.cfg.SOLVER.EPOCH,
                                                      initial_epoch=initial_epoch,
                                                      callbacks=[tensorboard, checkpoint,
                                                                 lr_scheduler, early_stopping])
            self.train_model.save_weights(os.path.join(checkpoint_dir, 'trained_yolo_model_stage_2.h5'))
        print('\n----------- end to train -----------\n')