# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 下午3:20
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : get_dr_txt.py
# @Software: PyCharm


"""
    这是不同YOLO模型生成测试数据集每张图片检测结果txt文件的脚本
"""

import os
import sys
import argparse
import colorsys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from config.config import get_cfg
from utils.model_utils import get_anchors
from utils.model_utils import get_classes
from utils.dataset_utils import letterbox_image

parser = argparse.ArgumentParser(description='get_gt_txt parameters')
parser.add_argument('--dataset_dir', type=str,help="voc dataset dir")
parser.add_argument('--config_file_path', type=str,help="voc dataset dir")
parser.add_argument('--ext', type=str,help="Image's ext such as .jpg, .png")
parser.add_argument('--h', type=int)
parser.add_argument('--w', type=int)
parser.add_argument('--images_optional_flag', action='store_true', default=False)
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

    def detect_image(self, image_id, image):
        '''
        这是利用YOLO模型对图像检测并生成每张图像检测结果txt文件的函数
        Args:
            image_id: 图像编号
            image: 输入图像
        Returns:
        '''
        # 检测图像并保存图像检测结果
        dr_txt_path = os.path.join(self.detection_result_dir,image_id+".txt")
        with open(dr_txt_path,"w") as g:
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
            #input_image_shape = np.expand_dims(np.array([h,w], dtype='float32'), 0)
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

            #print(len(out_classes),len(out_boxes),len(out_scores))
            for i, c in enumerate(out_classes):
                predicted_class = self.classes_names[int(c)]
                score = str(out_scores[i])
                top, left, bottom, right = out_boxes[i]
                #print(predicted_class,score,top, left, bottom, right)
                g.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

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
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
        return image

def run_main():
    """
       这是主函数
    """
    #  初始化YOLO模型训练阶段的超参数
    cfg = get_cfg()
    cfg.merge_from_file(os.path.abspath(args.config_file_path))
    cfg.merge_from_list(args.opts)

    # 初始化YOLO模型，用于计算mAP
    image_size = (args.h,args.w)
    yolo = mAP_YOLO(cfg,image_size)

    # 默认为False，不写入图片
    images_optional_flag = args.images_optional_flag

    input_dir = os.path.abspath("./mAP/input")
    detect_result_dir = os.path.join(input_dir, "detection-results")
    image_result_dir = os.path.join(input_dir,'images')
    if not os.path.exists(detect_result_dir):
        os.makedirs(detect_result_dir)
    if images_optional_flag:
        if not os.path.exists(image_result_dir):
            os.makedirs(image_result_dir)

    # 初始化测试数据集txt文件
    dataset_dir = os.path.abspath(args.dataset_dir)
    test_txt_path = os.path.join(dataset_dir,"ImageSets","Main","val.txt")
    ext = args.ext
    with open(test_txt_path,"r") as f:
        for line in f.readlines():
            image_id = line.strip()
            image_path = os.path.join(dataset_dir,"JPEGImages",image_id+ext)
            image = Image.open(image_path)
            image = yolo.detect_image(image_id,image)
            if images_optional_flag:
                image.save(os.path.join(image_result_dir,image_id+ext))
    print("Finished Detecting Test Dataset, Detection Result Conversion Completed!")

if __name__ == '__main__':
    run_main()