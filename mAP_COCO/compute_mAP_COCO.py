# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 下午3:31
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : compute_mAP_COCO.py
# @Software: PyCharm

import os
import sys
import json
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(description='get_gt_json parameters')
parser.add_argument('--gt_json_path', type=str)
parser.add_argument('--dr_json_path', type=str)
args = parser.parse_args()

def get_img_id(json_path):
    '''
    这是从JSON文件中获取图像id的函数
    Args:
        json_path: JSON文件路径
    Returns:
    '''
    ls = []
    annos = json.load(open(json_path,'r'))
    for anno in annos['annotations']:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def run_main():
    """
    这是主函数
    """
    gt_json_path = os.path.abspath(args.gt_json_path)
    dr_json_path = os.path.abspath(args.dr_json_path)

    coco_gt = COCO(gt_json_path)
    coco_dr = COCO(dr_json_path)
    img_ids = get_img_id(dr_json_path)
    img_ids = sorted(img_ids)
    coco_eval = COCOeval(coco_gt,coco_dr,'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()                #评价
    coco_eval.accumulate()              #积累
    coco_eval.summarize()               #总结

if __name__ == '__main__':
    run_main()