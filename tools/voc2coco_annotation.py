# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午4:56
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : voc2coco_annotation.py
# @Software: PyCharm

"""
    这是将VOC格式数据集转换成YOLO格式的脚本
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_utils import get_classes

parser = argparse.ArgumentParser(description='VOC2COCO_annotation parameters')
parser.add_argument('--voc_dataset_dir', type=str,default="./model_data/dataset/VOC2012")
parser.add_argument('--coco_dataset_dir',type=str,default="./model_data/COCO/VOC2012/")
parser.add_argument('--classes_path',type=str,default="./model_data/voc_classes.txt")
parser.add_argument('--ext',type=str,default=".jpg")
args = parser.parse_args()

def is_contain_object(xml_path):
    '''
    这是判断XML文件中是否包含目标的函数
    Args:
        xml_path: XML文件路径
    Returns:
    '''
    tree = ET.parse(xml_path)
    return len(tree.findall('object'))

def convert_annotation(annotation_path,list_file,classes):
    '''
    这是VOC格式数据集目标框标注转换为COCO数据集格式的目标框，并写入对应文件
    Args:
        annotation_path: xml标签文件
        list_file: COCO数据集文件
        classes: 目标分类数组
    Returns:
    '''
    # 解析XML标签文件
    in_file = open(annotation_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    # 遍历所有的目标
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

def voc2coco_annotation(voc_dataset_dir,coco_anotation_dir,choices,classes,ext='.jpg'):
    '''
    这是将VOC数据集格式标签转化为COCO数据集格式标签的函数
    Args:
        voc_dataset_dir: VOC数据集目录
        coco_anotation_dir: COCO数据集标签目录
        choices: 数据集类型选择
        classes: 目标分类数组
        ext: 图片后缀，默认为'.jpg'
    Returns:
    '''
    # 遍历每个类型数据集
    for choice in choices:
        # 初始化COCO数据集格式的标签文件路径
        coco_anotation_path = os.path.join(coco_anotation_dir, "%s.txt" % (choice))
        with open(coco_anotation_path, "w") as g:
            image_paths = []                    # 图像路径
            annotation_paths = []               # XML文件路径
            # 遍历每个VOC数据集，获取每种类型数据集的所有图像路径和XML文件路径
            txt_path = os.path.join(voc_dataset_dir, "ImageSets", "Main", choice + ".txt")
            with open(txt_path, "r") as f:
                for line in f.readlines():
                    #image_paths.append(os.path.join(voc_dataset_dir, dataset, "JPEGImages",line.strip()+".png"))
                    image_paths.append(os.path.join(voc_dataset_dir,"JPEGImages", line.strip() + ext))
                    annotation_paths.append(os.path.join(voc_dataset_dir,"Annotations",line.strip()+".xml"))
            # 构造COCO数据集格式的标签文件
            for image_path,annotation_path in zip(image_paths,annotation_paths):
                if is_contain_object(annotation_path):                          # XML文件包含目标
                    g.write(image_path)                                         # 写入图像路径
                    convert_annotation(annotation_path, g,classes)              # 写入图像对应的目标框信息
                    g.write('\n')

def run_main():
    """
       这是主函数
    """
    voc_dataset_dir = os.path.abspath(args.voc_dataset_dir)
    coco_dataset_dir = os.path.abspath(args.coco_dataset_dir)  # COCO数据集标注目录
    if not os.path.exists(coco_dataset_dir):
        os.makedirs(coco_dataset_dir)
    choices = ["train", "val", "trainval"]  # 数据集种类，训练、验证、训练验证和测试文件
    ext = args.ext
    classes_path = os.path.abspath(args.classes_path)
    classes = list(get_classes(classes_path))
    print(classes)
    voc2coco_annotation(voc_dataset_dir, coco_dataset_dir, choices, classes,ext)

if __name__ == '__main__':
    run_main()