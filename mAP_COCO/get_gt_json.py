# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 下午3:30
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : get_gt_json.py
# @Software: PyCharm

"""
    这是生成测试数据集每张图像中真实目标及其定位信息json文件的脚本
"""

import os
import cv2
import sys
import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.model_utils import NpEncoder
from utils.model_utils import get_classes

classes_path = os.path.abspath("../model_data/voc_classes.txt")

parser = argparse.ArgumentParser(description='get_gt_json parameters')
parser.add_argument('--dataset_dir', type=str,help="voc dataset dir")
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--ext', type=str,default='.jpg')
parser.add_argument('--classes_path', type=str)
args = parser.parse_args()

def is_contain_object(xml_path):
    '''
    这是判断XML文件中是否包含目标标签的函数
    Args:
        xml_path: XML文件路径
    Returns:
    '''
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    return len(root.findall('object')) > 0

def parse_xml(xml_path):
    '''
    这是解析VOC数据集XML标签文件，获取每个目标分类与定位的函数
    Args:
        xml_path: XML标签文件路径
    Returns:
    '''
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    # 遍历所有目标
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        left = bndbox.find('xmin').text
        top = bndbox.find('ymin').text
        right = bndbox.find('xmax').text
        bottom = bndbox.find('ymax').text
        objects.append([obj_name, float(left), float(top), float(right), float(bottom)])
    return objects

def run_main():
    """
       这是主函数
    """
    # 初始化目标分类名称指点
    classes_path = os.path.abspath(args.class_path)
    classes = get_classes(classes_path)
    cls2num_dict = dict(zip(classes, np.arange(len(classes))))

    # 初始化JSON文件路径
    dataset_name = args.dataset_name
    input_dir = os.path.abspath("./input/{0}".format(dataset_name))
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 初始化测试数据集txt文件
    dataset_dir = os.path.abspath(args.dataset_dir)
    ext = args.ext
    gt_result = {}
    gt_result_json_path = os.path.join(input_dir, 'gt_result.json')
    image_array = []
    annotation_array = []
    img_cnt = 0
    anno_cnt = 0
    with open(gt_result_json_path, 'w+') as f:
        test_txt_path = os.path.join(dataset_dir, "ImageSets", "Main", "val.txt")
        image_ids = []
        with open(test_txt_path,'r') as g:
            for line in g.readlines():
                image_ids.append(line.strip())

        # 生成测试集的groundtruth的分类与定位的json文件
        annotation_dir = os.path.join(dataset_dir,"Annotations")
        image_dir = os.path
        for image_id in image_ids:
            xml_path = os.path.join(annotation_dir, image_id+".xml")
            image_path = os.path.join(image_dir, image_id+ext)
            image = cv2.imread(image_path)
            h,w,c = np.shape(image)
            image_array.append({'file_name': image_id+ext, 'id': img_cnt,'width':w,'height':h})
            if is_contain_object(xml_path):
                objects = parse_xml(xml_path)
                for obj in objects:
                    cls_name,xmin,ymin,xmax,ymax = obj
                    w = int(xmax)-int(xmin)
                    h = int(ymax)-int(ymin)
                    annotation_array.append({'image_id':img_cnt,
                                             'iscrowd':0,
                                             'bbox':[int(xmin),int(ymin),w,h],
                                             'area':int(w*h),
                                             "category_id":cls2num_dict[cls_name],
                                             'id':anno_cnt})
                    anno_cnt += 1
            img_cnt += 1
        gt_result['images'] = image_array
        gt_result["annotations"] = annotation_array
        gt_result["categories"] = [{"id":id,"name":cls_name} for cls_name,id in cls2num_dict.items()]
        gt_result_json_data = json.dumps(gt_result,indent=4,separators=(',', ': '), cls=NpEncoder)
        print(gt_result_json_data)
        f.write(gt_result_json_data)
        print("Test Dataset GroundTruth Result Conversion Completed!")

if __name__ == '__main__':
    run_main()