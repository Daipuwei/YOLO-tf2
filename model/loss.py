# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 下午11:19
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : loss.py
# @Software: PyCharm

"""
    这是YOLO模型的损失函数的定义脚本，目前目标分类损失支持smooth Label;
    目标定位损失支持均方差损失、GIOU Loss、DIOU Loss和CIOU Loss；
"""

import math
import tensorflow as tf
from tensorflow.keras import backend as K

# ---------------------------------------------------#
#   平滑标签
# ---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # ---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    # ---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
# ---------------------------------------------------#
def box_iou(b_true, b_pred):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b_true = K.expand_dims(b_true, -2)
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh / 2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b_pred = K.expand_dims(b_pred, 0)
    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh / 2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    # 计算重合面积
    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    iou = intersect_area / (b_true_area + b_pred_area - intersect_area)

    return iou

def box_giou(b_true, b_pred):
    """
    Calculate GIoU loss on anchor boxes
    Reference Paper:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        https://arxiv.org/abs/1902.09630
    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    Returns
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh / 2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh / 2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
    giou = K.expand_dims(giou, -1)

    return giou

def box_diou(b_true, b_pred,use_ciou_loss=False):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh / 2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half
    # 求出真实框左上角右下角
    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh / 2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b1_area + b_pred_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    # 计算中心的差距
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # 计算对角线距离
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    diou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())

    if use_ciou_loss:
        v = 4 * K.square(tf.math.atan2(b_true_wh[..., 0], K.maximum(b_true_wh[..., 1], K.epsilon()))
                         - tf.math.atan2(b_pred_wh[..., 0],K.maximum(b_pred_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
        # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
        #          to match related description for equation (12) in original paper
        #
        #
        #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
        #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
        #
        #          The dominator w^2+h^2 is usually a small value for the cases
        #          h and w ranging in [0; 1], which is likely to yield gradient
        #          explosion. And thus in our implementation, the dominator
        #          w^2+h^2 is simply removed for stable convergence, by which
        #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
        #          is still consistent with Eqn. (12).
        v = v * tf.stop_gradient(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1])
        alpha = v / K.maximum((1.0 - iou + v), K.epsilon())
        diou = diou - alpha * v

    diou = K.expand_dims(diou, -1)
    diou = tf.where(tf.math.is_nan(diou), tf.zeros_like(diou), diou)
    return diou


# ---------------------------------------------------#
#   loss值计算
# ---------------------------------------------------#
def yolo_loss(args, anchors,num_classes,ignore_threshold=.5,label_smoothing=0.1,
              use_giou_loss=False,use_diou_loss=False,use_ciou_loss=False,normalize=True,model_name='yolov3'):
    # 根据不同yolo模型初始化不同anchor掩膜和输出层数
    if model_name == "yolov3":                              # yolov3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = 3
    elif model_name == 'yolov3-spp':                        # yolov3-spp
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = 3
    elif model_name == 'yolov4':                            # yolov4
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = 3
    elif model_name == 'yolov4-csp':                        # yolov4-csp
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = 3
    elif model_name == 'yolov4-p5':                         # yolov4-p5
        anchor_mask = [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
        num_layers = 3
    elif model_name == 'yolov4-p6':                         # yolov4-p6
        anchor_mask = [[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
        num_layers = 4
    elif model_name == 'yolov4-p7':                         # yolov4-p7
        anchor_mask = [[16, 17, 18, 19], [12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
        num_layers = 5
    elif model_name == 'yolov3-tiny':                       # yolov3-tiny
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        num_layers = 2
    elif model_name == 'yolov4-tiny':                       # yolov4-tiny
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        num_layers = 2
    else:                                                   # 默认为yolov3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = 3

    #   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # 根据不同yolo模型初始化输入尺度和网格尺度
    if model_name == "yolov3":                              # yolov3
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    elif model_name == 'yolov3-spp':                        # yolov3-spp
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    elif model_name == 'yolov4':                            # yolov4
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    elif model_name == 'yolov4-csp':                        # yolov4-csp
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    elif model_name == 'yolov4-p5':                         # yolov4-p5
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    elif model_name == 'yolov4-p6':                         # yolov4-p6
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*64, K.dtype(y_true[0]))
    elif model_name == 'yolov4-p7':                         # yolov4-p7
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*128, K.dtype(y_true[0]))
    elif model_name == 'yolov3-tiny':                       # yolov3-tiny
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    elif model_name == 'yolov4-tiny':                       # yolov4-tiny
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    else:                                                   # 默认为yolov3
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[l])) for l in range(num_layers)]

    loss = 0
    num_pos = 0
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    for l in range(num_layers):
        # -----------------------------------------------------------#
        #   以第一个特征层(m,13,13,3,85)为例子
        #   取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        # -----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:             # 使用平滑标签
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # -----------------------------------------------------------#
        #   将yolo_outputs的特征层输出进行处理、获得四个返回值
        #   grid为网格坐标
        #   raw_pred为尚未处理的预测结果
        #   pred_xy为解码后的中心坐标
        #   pred_wh为解码后的宽高坐标
        # -----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        # pred_box是解码后的预测的box的位置
        pred_box = K.concatenate([pred_xy, pred_wh])

        # -----------------------------------------------------------#
        #   找到负样本群组，第一步是创建一个数组，[]
        # -----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # 取出n个真实框：n,4
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # -----------------------------------------------------------#
            #   计算预测框与真实框的iou
            #   pred_box为预测框的坐标
            #   true_box为真实框的坐标
            #   iou为预测框和真实框的iou
            # -----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)
            #   best_iou为每个特征点与真实框的最大重合程度
            best_iou = K.max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   判断预测框和真实框的最大iou小于ignore_thresh
            #   则认为该预测框没有与之对应的真实框
            #   该操作的目的是：
            #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
            #   不适合当作负样本，所以忽略掉。
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_threshold, K.dtype(true_box)))
            return b + 1, ignore_mask

        # 在这个地方进行一个循环、循环是对每一张图片进行的
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # ignore_mask用于提取出作为负样本的特征点
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # 真实框越大，比重越小，小框的比重更大。
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # ------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   该操作的目的是：
        #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
        #   不适合当作负样本，所以忽略掉。
        # ------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        # 根据不同参数选择不同定位损失
        if use_giou_loss:                         # 计算GIOU损失
            raw_true_box = y_true[l][..., 0:4]
            giou = box_giou(raw_true_box, pred_box)
            giou_loss = object_mask * box_loss_scale * (1 - giou)
            giou_loss = K.sum(giou_loss)
            location_loss = giou_loss
        elif use_diou_loss:                       # 计算DIOU损失
            raw_true_box = y_true[l][..., 0:4]
            diou = box_diou(pred_box, raw_true_box, use_ciou_loss=False)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            location_loss = diou_loss
        elif use_ciou_loss:                       # 计算CIOU损失
            raw_true_box = y_true[l][..., 0:4]
            ciou = box_diou(pred_box, raw_true_box,use_ciou_loss=True)
            ciou_loss = object_mask * box_loss_scale * (1 - ciou)
            location_loss = ciou_loss
        else:                                  # YOLO v3边界框定位损失
            # Standard YOLOv3 location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
            raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            xy_loss = K.sum(xy_loss)
            wh_loss = K.sum(wh_loss)
            location_loss = xy_loss + wh_loss
        location_loss = K.sum(location_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        # 计算正样本数量
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += location_loss + confidence_loss + class_loss
    loss = K.expand_dims(loss, axis=-1)
    # 计算YOLO模型损失
    if normalize:
        loss = loss / num_pos
    else:
        loss = loss / mf
    return loss