# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:23
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : dataset_utils.py
# @Software: PyCharm

"""
    这是YOLO模型数据集
"""

import cv2
import numpy as np

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from utils.model_utils import get_classes
from utils.model_utils import get_anchors

def resize_keep_aspect_ratio(image_src, dst_size, value=[128, 128, 128]):
    '''
    这是opencv将源图像扩充边界成正方形，并完成图像尺寸变换
    Args:
        image_src: 源图像
        dst_size: 缩放尺寸
        value: 填充像素值
    Returns:
    '''
    # 获取源图像和目标图像的尺寸
    src_h, src_w, _ = np.shape(image_src)
    dst_h, dst_w = dst_size

    # 首先确定哪个方向进行填充
    if src_h < src_w:  # 在h方向进行填充
        delta = src_w - src_h  # 计算需要填充的像素个数，然后均分到上下两侧
        top = int(delta // 2)
        down = delta - top
        left = 0
        right = 0
    else:  # 在w方向进行填充
        delta = src_h - src_w  # 计算需要填充的像素个数，然后均分到左右两侧
        top = 0
        down = 0
        left = int(delta // 2)
        right = delta - left
    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_src, top, down, left, right, borderType, None, value)
    image_dst = cv2.resize(image_dst, dst_size)
    return image_dst

def letterbox_image(image, size):
    '''
    这是PIL将源图像扩充边界成正方形，并完成图像尺寸变换
    Args:
        image: 图像
        size: 缩放尺寸
    Returns:
    '''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Dataset(object):

    def __init__(self,dataset_path,classes_path,anchors_path,batch_size,target_size,
                 max_boxes_num=20,use_mosaic=False,random=True,model_name='yolov3'):
        '''
        这是目标检测数据集初始化类
        Args:
            dataset_path: COCO格式的数据集txt地址
            classes_path: 目标分类txt地址
            anchors_path: 模版框txt地址
            batch_size: 小批量数规模
            target_size: 目标尺寸
            max_boxes_num: 最大目标框个数,默认为20
            use_mosaic: 是否使用mosaic数据增强，默认为False
            random: 是否进行随机数据增强标志量，默认为True
            model_name: 模型名称，默认为‘yolov3’
        '''
        self.dataset_path = dataset_path
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.target_size = target_size
        self.max_boxes_num = max_boxes_num
        self.use_mosaic = use_mosaic
        self.random = random
        self.model_name = model_name

        self.annotation_lines = []
        with open(self.dataset_path, 'r') as f:
            for line in f.readlines():
                self.annotation_lines.append(line)
        self.annotation_lines = np.array(self.annotation_lines)
        self.annotation_lines = np.random.permutation(self.annotation_lines)
        self.size = len(self.annotation_lines)
        self.batch_size = batch_size
        self.iter_num = self.size // self.batch_size
        if self.size % self.batch_size != 0:
            self.iter_num += 1

        # 初始化anchors与classes
        self.anchors = get_anchors(self.anchors_path)
        self.classes_names = get_classes(self.classes_path)
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.classes_names)

        # 初始化相关数据增强参数
        self.jitter = 0.3
        self.hue=.1
        self.sat=1.5
        self.val=1.5

    def get_batch_data_with_mosaic(self,batch_annotation_lines):
        '''
        这是获取批量图像及其标签并使用mosaic数据增强的函数
        Args:
            batch_annotation_lines: 批量yolo数据集格式标注
        Returns:
        '''
        batch_image_data = []
        batch_boxes = []
        size = len(batch_annotation_lines)
        for start in np.arange(0,len(batch_annotation_lines),4):
            end = int(np.min([start+4,size]))
            _batch_annotation_lines = batch_annotation_lines[start:end]
            image_data,box_data = self.get_random_data_with_mosaic(_batch_annotation_lines)
            batch_image_data.append(image_data)
            batch_boxes.append(box_data)
        batch_image_data = np.array(batch_image_data)
        batch_boxes = np.array(batch_boxes)
        return batch_image_data,batch_boxes

    def get_random_data_with_mosaic(self,batch_lines):
        """
        这是4张图像及其目标标签，并对图像法进行mosaic数据增强操作的函数
        :param batch_lines: 4张yolo格式数据
        :return:
        """
        h, w = self.target_size
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]

        # 批量图像可能不足4张，随机补充
        size = len(batch_lines)
        if size < 4:
            dif = 4 - len(batch_lines)
            _batch_line = [line for line in batch_lines]
            for i in np.arange(dif):
                random_index = np.random.randint(0,size)
                _batch_line.append(batch_lines[random_index])
            batch_lines = np.array(_batch_line)
        # 便利所有图像，加载真实标签
        for line in batch_lines:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            hue = rand(-self.hue, self.hue)
            sat = rand(1, self.sat) if rand() < .5 else 1 / rand(1, self.sat)
            val = rand(1, self.val) if rand() < .5 else 1 / rand(1, self.val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 归并边界框
        merge_bbox = self.merge_bboxes(box_datas,cutx,cuty)
        #print(np.shape(merge_bbox))
        bbox = np.zeros((self.max_boxes_num, 5))
        if len(merge_bbox) != 0:
            if len(merge_bbox) > self.max_boxes_num:
                merge_bbox = merge_bbox[:self.max_boxes_num]
            bbox[:len(merge_bbox)] = merge_bbox
        return new_image,bbox

    def merge_bboxes(self,bbox_data,cutx,cuty):
        '''
         这是mosaic数据增强中对4张图片的边界框标签进行合并的函数
        Args:
            bbox_data: 边界框标签数组
            cutx: x坐标轴分界值
            cuty: y坐标轴分界值
        Returns:
        '''

        merge_bbox = []
        for i,bboxes in enumerate(bbox_data):
            if bboxes is not None:
                for box in bboxes:
                    tmp_box = []
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    if i == 0:
                        if y1 > cuty or x1 > cutx:
                            continue
                        if y2 >= cuty and y1 <= cuty:
                            y2 = cuty
                            if y2 - y1 < 5:         # 相差过小则放弃
                                continue
                        if x2 >= cutx and x1 <= cutx:
                            x2 = cutx
                            if x2 - x1 < 5:         # 相差过小则放弃
                                continue

                    if i == 1:
                        if y2 < cuty or x1 > cutx:
                            continue

                        if y2 >= cuty and y1 <= cuty:
                            y1 = cuty
                            if y2 - y1 < 5:         # 相差过小则放弃
                                continue

                        if x2 >= cutx and x1 <= cutx:
                            x2 = cutx
                            if x2 - x1 < 5:         # 相差过小则放弃
                                continue

                    if i == 2:
                        if y2 < cuty or x2 < cutx:
                            continue

                        if y2 >= cuty and y1 <= cuty:
                            y1 = cuty
                            if y2 - y1 < 5:         # 相差过小则放弃
                                continue

                        if x2 >= cutx and x1 <= cutx:
                            x1 = cutx
                            if x2 - x1 < 5:         # 相差过小则放弃
                                continue

                    if i == 3:
                        if y1 > cuty or x2 < cutx:
                            continue

                        if y2 >= cuty and y1 <= cuty:
                            y2 = cuty
                            if y2 - y1 < 5:         # 相差过小则放弃
                                continue

                        if x2 >= cutx and x1 <= cutx:
                            x1 = cutx
                            if x2 - x1 < 5:         # 相差过小则放弃
                                continue
                    tmp_box.append(x1)
                    tmp_box.append(y1)
                    tmp_box.append(x2)
                    tmp_box.append(y2)
                    tmp_box.append(box[-1])
                    merge_bbox.append(tmp_box)
        del bbox_data
        return np.array(merge_bbox)

    def get_batch_data(self,batch_annotation_lines):
        '''
        这是获取批量图像及其目标框标签的函数，不使用mosaic数据增强
        Args:
            batch_annotation_lines: 批量yolo数据集格式标注
        Returns:
        '''
        batch_images = []
        batch_boxes = []
        for annotation_line in batch_annotation_lines:
            image,box_data = self.get_random_data(annotation_line)
            batch_images.append(image)
            batch_boxes.append(box_data)
        batch_images = np.array(batch_images)
        batch_boxes = np.array(batch_boxes)

        return batch_images,batch_boxes

    def get_random_data(self,line):
        '''
        这是获取图像及其目标标签，并对图像法进行实时数据增强操作的函数
        Args:
            line: yolo格式数据
        Returns:
        '''
        lines =line.split()
        image = Image.open(lines[0])
        iw, ih = image.size
        h, w = self.target_size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in lines[1:]])

        if not self.random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32) / 255

            # correct boxes
            box_data = np.zeros((self.max_boxes_num, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
                if len(box) > self.max_boxes_num: box = box[:self.max_boxes_num]
                box_data[:len(box)] = box
            return image_data, box_data

        # resize image
        new_ar = w / h * rand(1 - self.jitter, 1 + self.jitter) / rand(1 - self.jitter, 1 + self.jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-self.hue, self.hue)
        sat = rand(1, self.sat) if rand() < .5 else 1 / rand(1, self.sat)
        val = rand(1, self.val) if rand() < .5 else 1 / rand(1, self.val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((self.max_boxes_num, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > self.max_boxes_num:
                box = box[:self.max_boxes_num]
            box_data[:len(box)] = box

        return image_data, box_data

    # ---------------------------------------------------#
    #   读入xml文件，并输出y_true
    # ---------------------------------------------------#
    def preprocess_true_boxes(self,true_boxes):
        '''
        这是根据真实标签转换成不同yolo预测输出的函数
        Args:
            true_boxes: 真实目标框标签
        Returns:

        '''
        assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'
        # -----------------------------------------------------------#
        #   获得框的坐标和图片的大小
        # -----------------------------------------------------------#
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(self.target_size, dtype='int32')

        # 根据不同yolo模型初始化不同anchor掩膜、网格尺寸和输出层数
        if self.model_name == 'yolov3':                     # yolov3
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_layers = 3
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov3-spp':               # yolov3-spp
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_layers = 3
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov4':                   # yolov4
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_layers = 3
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov4-csp':               # yolov4-csp
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_layers = 3
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov4-p5':                # yolov4-p5
            anchor_mask = [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
            num_layers = 3
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov4-p6':                # yolov4-p6
            anchor_mask = [[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
            num_layers = 4
            grid_shapes = [input_shape // {0: 64, 1: 32, 2: 16, 3: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov4-p7':                # yolov4-p7
            anchor_mask = [[16, 17, 18, 19], [12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
            num_layers = 5
            grid_shapes = [input_shape // {0:128, 1: 64, 2: 32, 3: 16, 4: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'poly-yolo':                # poly-yolo(v3)
            anchor_mask = [[0,1,2,3,4,5,6,7,8]]
            num_layers = 1
            grid_shapes = [input_shape // {0: 8}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov3-tiny':              # yolov3-tiny
            anchor_mask = [[3, 4, 5], [0, 1, 2]]
            num_layers = 2
            grid_shapes = [input_shape // {0: 32, 1: 16}[l] for l in range(num_layers)]
        elif self.model_name == 'yolov4-tiny':              # yolov4-tiny
            anchor_mask = [ [3, 4, 5], [0, 1, 2]]
            num_layers = 2
            grid_shapes = [input_shape // {0: 32, 1: 16}[l] for l in range(num_layers)]
            print(grid_shapes)
        else:                                               # 默认为yolov3
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_layers = 3
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

        # -----------------------------------------------------------#
        #   通过计算获得真实框的中心和宽高
        #   中心点(m,n,2) 宽高(m,n,2)
        # -----------------------------------------------------------#
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # -----------------------------------------------------------#
        #   将真实框归一化到小数形式
        # -----------------------------------------------------------#
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # m为图片数量，grid_shapes为网格的shape
        m = true_boxes.shape[0]
        #grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        # -----------------------------------------------------------#
        #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
        # -----------------------------------------------------------#
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes),
                           dtype='float32') for l in range(num_layers)]

        # -----------------------------------------------------------#
        #   [9,2] -> [1,9,2]
        # -----------------------------------------------------------#
        anchors = np.expand_dims(self.anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes

        # -----------------------------------------------------------#
        #   长宽要大于0才有效
        # -----------------------------------------------------------#
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            # 对每一张图进行处理
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # -----------------------------------------------------------#
            #   [n,2] -> [n,1,2]
            # -----------------------------------------------------------#
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            # -----------------------------------------------------------#
            #   计算所有真实框和先验框的交并比
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            # -----------------------------------------------------------#
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # -----------------------------------------------------------#
            #   维度是[n,] 感谢 消尽不死鸟 的提醒
            # -----------------------------------------------------------#
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                # -----------------------------------------------------------#
                #   找到每个真实框所属的特征层
                # -----------------------------------------------------------#
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        # -----------------------------------------------------------#
                        #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                        # -----------------------------------------------------------#
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        # -----------------------------------------------------------#
                        #   k指的的当前这个特征点的第k个先验框
                        # -----------------------------------------------------------#
                        k = anchor_mask[l].index(n)
                        # -----------------------------------------------------------#
                        #   c指的是当前这个真实框的种类
                        # -----------------------------------------------------------#
                        c = true_boxes[b, t, 4].astype('int32')
                        # -----------------------------------------------------------#
                        #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                        #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                        #   1代表的是置信度、80代表的是种类
                        # -----------------------------------------------------------#
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true

    def generator(self):
        '''
        这是数据生成器定义函数
        Returns:
        '''

        while True:
            # 随机打乱数据集
            self.annotation_lines = np.random.permutation(self.annotation_lines)
            for start in np.arange(0,self.size,self.batch_size):
                end = int(np.min([start+self.batch_size,self.size]))
                batch_annotation_lines = self.annotation_lines[start:end]
                if self.use_mosaic:
                    batch_images,batch_boxes = self.get_batch_data_with_mosaic(batch_annotation_lines)
                else:
                    batch_images, batch_boxes = self.get_batch_data(batch_annotation_lines)

                # 对box数组进行处理，生成符合YOLO v4模型输出的标签
                batch_y_true = self.preprocess_true_boxes(batch_boxes)
                batch_loss = np.zeros(len(batch_images))

                yield [batch_images,*batch_y_true],batch_loss