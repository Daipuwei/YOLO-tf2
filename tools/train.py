# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:33
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : train.py
# @Software: PyCharm

"""
    这是YOLO模型的训练脚本
"""

import os
import sys
import argparse
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import get_cfg
from utils.train_utils import Trainer

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

parser = argparse.ArgumentParser(description='YOLO training parameters')
parser.add_argument('--config_file_path', type=str,default="./config/VOC/yolov3.yaml",help="YOLO model config file path")
parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
args = parser.parse_args()

def run_main():
    """
    这是主函数
    """
    #  初始化YOLO模型训练阶段的超参数
    cfg = get_cfg()
    cfg.merge_from_file(os.path.abspath(args.config_file_path))
    if args.opts is not None:           # 可以用来覆盖yaml部分参数，例如模型权重路径，anchor数组txt文件路径和目标分类txt文件路径等
        cfg.merge_from_list(args.opts)

    # 初始化YOLO模型训练类，并进行模型训练
    yolo_trainer = Trainer(cfg)
    yolo_trainer.train()

if __name__ == '__main__':
    run_main()