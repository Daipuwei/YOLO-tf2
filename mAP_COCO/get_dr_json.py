# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 下午3:30
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : get_dr_json.py
# @Software: PyCharm

""""
    这是利用YOLO v3模型对测试数据集进行检测，
    生成测试数据集每张图片检测结果json文件的脚本
"""

import os
import sys
import json
import argparse
import colorsys
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.config import get_cfg
from utils.model_utils import NpEncoder
from utils.model_utils import get_anchors
from utils.model_utils import get_classes
from utils.dataset_utils import letterbox_image

parser = argparse.ArgumentParser(description='get_gt_json parameters')
parser.add_argument('--dataset_dir', type=str,help="voc dataset dir")
parser.add_argument('--config_file_path', type=str,help="yolo model yaml file path")
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--ext', type=str,help="Image's ext such as .jpg, .png")
parser.add_argument('--images_optional_flag', action='store_true', default=False)
parser.add_argument('--h', type=int)
parser.add_argument('--w', type=int)
parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
args = parser.parse_args()

class mAP_YOLO(object):

    def __init__(self,cfg,image_size):
        '''
        计算mAP的YOLO模型类的初始化函数
        Args:
            cfg: 参数字典
        '''
        # 初始化相关参数
        self.cfg = cfg
        self.model_name = cfg.MODEL.MODEL_NAME
        self.image_size = image_size
        self.font_path = os.path.abspath(cfg.FONT_PATH)
        self.detection_result_dir = os.path.abspath("./mAP/input/detection-results/")

        # 初始化目标框与模版框
        self.classes_names = get_classes(os.path.abspath(cfg.DATASET.CLASSES_PATH))
        self.anchors = get_anchors(os.path.abspath(cfg.DATASET.ANCHORS_PATH))
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.classes_names)

        self.generate()

    def generate(self):
        '''
        这是生成YOLO v3检测计算图，并对检测结果进行解码的函数
        '''
        # 初始化不同YOLO模型
        if self.model_name == "yolov3":  # yolov3
            from model.yolov3 import build_yolov3_eval
            self.yolo_eval_model = build_yolov3_eval(self.cfg)
        elif self.model_name == 'yolov3-spp':  # yolov3-spp
            from model.yolov3 import build_yolov3_eval
            self.yolo_eval_model = build_yolov3_eval(self.cfg)
        elif self.model_name == 'yolov4':  # yolov4
            from model.yolov4 import build_yolov4_eval
            self.yolo_eval_model = build_yolov4_eval(self.cfg)
        elif self.model_name == 'yolov4-csp':  # yolov4-csp
            from model.yolov4_csp import build_yolov4_csp_eval
            self.yolo_eval_model = build_yolov4_csp_eval(self.cfg)
        elif self.model_name == 'yolov4-p5':  # yolov4-p5
            from model.yolov4_p5 import build_yolov4_p5_eval
            self.yolo_eval_model = build_yolov4_p5_eval(self.cfg)
        elif self.model_name == 'yolov4-p6':  # yolov4-p6
            from model.yolov4_p6 import build_yolov4_p6_eval
            self.yolo_eval_model = build_yolov4_p6_eval(self.cfg)
        elif self.model_name == 'yolov4-p7':  # yolov4-p7
            from model.yolov4_p7 import build_yolov4_p7_eval
            self.yolo_eval_model = build_yolov4_p7_eval(self.cfg)
        elif self.model_name == 'yolov3-tiny':  # yolov3-tiny
            from model.yolov3_tiny import build_yolov3_tiny_eval
            self.yolo_eval_model = build_yolov3_tiny_eval(self.cfg)
        elif self.model_name == 'yolov4-tiny':  # yolov4-tiny
            from model.yolov4_tiny import build_yolov4_eval
            self.yolo_eval_model = build_yolov4_eval(self.cfg)
        else:  # 默认为yolov3
            from model.yolov3 import build_yolov3_eval
            self.cfg.MODEL.USE_SPP = False
            self.yolo_eval_model = build_yolov3_eval(self.cfg)

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.classes_names), 1., 1.)
                      for x in range(len(self.classes_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

    def detect_image(self, image):
        '''
        这是利用YOLO模型对图像检测的函数
        Args:
            image: 输入图像
        Returns:
        '''
        # 检测图像并保存图像检测结果
        # 检测图像并保存图像检测结果
        detection_results = []
        # 调整图片使其符合输入要求
        if self.cfg.DATASET.LETTERBOX_IMAGE:
            boxed_image = letterbox_image(image, (self.image_size[1], self.image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.image_size[1], self.image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        outputs = self.yolo_eval_model.predict([image_data,input_image_shape])
        out_boxes, out_scores, out_classes = outputs

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # ---------------------------------------------------------#
        #   设置字体
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font=self.font_path,
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in enumerate(out_classes):
            predicted_class = self.classes_names[int(c)]
            score = str(out_scores[i])
            top, left, bottom, right = out_boxes[i]
            detection_results.append([predicted_class, int(c), float(score[:6]), int(left),
                                      int(top), int(right), int(bottom)])

            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, float(score))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i],outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=self.colors[c])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
        return image

def run_main():
    """
       这是主函数
    """
    # 初始化YOLO模型训练阶段的超参数
    cfg = get_cfg()
    cfg.merge_from_file(os.path.abspath(args.config_file_path))
    cfg.merge_from_list(args.opts)

    # 初始化目标分类名称指点
    classes_path = os.path.abspath(cfg.DATASET.CLASSES_PATH)
    classes = get_classes(classes_path)
    cls2num_dict = dict(zip(classes, np.arange(len(classes))))

    # 初始化YOLO模型
    image_size = (args.h, args.w)
    yolo = mAP_YOLO(cfg, image_size)

    ext = args.ext
    images_optional_flag = args.images_optional_flag                # 默认为False，不写入图片
    dataset_name = args.dataset_name
    input_dir = os.path.abspath("./input/{0}".format(dataset_name))
    image_result_dir = os.path.join(input_dir, 'images')
    if images_optional_flag:
        if not os.path.exists(image_result_dir):
            os.makedirs(image_result_dir)

    # 初始化测试数据集txt文件
    dataset_dir = os.path.abspath(args.dataset_dir)
    dr_result_json_path = os.path.join(input_dir, 'dr_result.json')
    dr_result = {}
    image_array = []
    annotation_array = []
    img_cnt = 0
    anno_cnt = 0
    with open(dr_result_json_path, 'w+') as f:
        test_txt_path = os.path.join(dataset_dir,"ImageSets","Main","val.txt")
        image_ids = []
        with open(test_txt_path,"r") as g:
            for line in g.readlines():
                image_id = line.strip()
                image_ids.append(image_id)
        for image_id in image_ids:
            image_path = os.path.join(dataset_dir,"JPEGImages",image_id+ext)
            image = Image.open(image_path)
            detect_results,image = yolo.detect_image(image)
            if images_optional_flag:
                image.save(os.path.join(image_result_dir,))
            image_array.append({'file_name': image_id + ".jpg", 'id': img_cnt, 'width': 960, 'height': 720})
            if len(detect_results) > 0:
                for _detect_result in detect_results:
                    cls_name,cls_num,score,xmin,ymin,xmax,ymax = _detect_result
                    w = xmax-xmin
                    h = ymax-ymin
                    annotation_array.append({'image_id': img_cnt,
                                             'iscrowd':0,
                                             'bbox': [int(xmin),int(ymin),w,h],
                                             'area':int(w*h),
                                             "category_id": cls_num,
                                             'id': anno_cnt,
                                             'score':score})
                    anno_cnt += 1
            else:
                continue
            img_cnt += 1
        dr_result['images'] = image_array
        dr_result["annotations"] = annotation_array
        dr_result["categories"] = [{"id": id, "name": cls_name} for cls_name, id in cls2num_dict.items()]
        dr_result_json_data = json.dumps(dr_result,indent=4,separators=(',', ': '),cls=NpEncoder)
        print(dr_result_json_data)
        f.write(dr_result_json_data)
        print("Finished Detecting Test Dataset, Detection Result Conversion Completed!")

if __name__ == '__main__':
    run_main()