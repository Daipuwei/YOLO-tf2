# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午4:55
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : kmeans.py
# @Software: PyCharm

"""
    这是利用Kmeans聚类算法生成指定数据集的anchor尺寸的脚本
"""

import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description='Kmeans parameters')
parser.add_argument('--cluster_number', type=int,default=9)
parser.add_argument('--dataset_path',type=str,default="./model_data/COCO/VOC/trainval.txt")
parser.add_argument('--yolo_anchors_path',type=str,default="./model_data/yolo_anchors_voc.txt")
args = parser.parse_args()

class YOLO_KMeans:

    def __init__(self, cluster_number, dataset_path):
        '''
        这是YOLO v3的K-Means聚类算法类的初始化函数
        Args:
            cluster_number: 聚类个数
            dataset_path: COCO格式数据集的txt文件路径
        '''
        self.cluster_number = cluster_number
        self.dataset_path = dataset_path

    def iou(self, boxes, clusters):
        '''
        这是计算IOU的函数
        Args:
            boxes: 目标框数组，shape：(n,2) 每个目标框组成为(w,h)
            clusters: 聚类中心(模版框)，shape：(self.cluster_number,2)
        Returns:
        '''
        # 初始化相关变量
        n = boxes.shape[0]
        k = self.cluster_number

        # 计算目标框面积
        box_area = boxes[:, 0] * boxes[:, 1]                                            # shape:(n,1)
        box_area = box_area.repeat(k)                                                   # 重复K次
        box_area = np.reshape(box_area, (n, k))                                         # 完成shape转化

        # 计算聚类中心（模版框）面积
        cluster_area = clusters[:, 0] * clusters[:, 1]                                  # shape：(k,1)
        cluster_area = np.tile(cluster_area, [1, n])                                    # Y轴不扩大，x轴扩大n被，变成(n*k,1)
        cluster_area = np.reshape(cluster_area, (n, k))                                 # 完成shape转换

        # 计算目标框与聚类中心(模版框)交集的w
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        # 计算目标框与聚类中心(模版框)交集的h
        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        # 计算目标框与聚类中心(模版框)之间IOU
        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        '''
        这是计算平均IOU的函数
        Args:
            boxes: 目标框数组，shape：(n,2) 每个目标框组成为(w,h)
            clusters: 聚类中心(模版框)，shape：(self.cluster_number,2)

        Returns:

        '''
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        '''
        这是YOLO v3用于自动生成模版框的K-Means聚类算法
        Args:
            boxes: 目标框
            k: 聚类个数
            dist: 距离类型
        Returns:
        '''
        box_number = boxes.shape[0]                     # 目标框个数
        distances = np.empty((box_number, k))           # 目标框与聚类中心距离初始化
        last_nearest = np.zeros((box_number,))          # 最后一次最近距离下标
        np.random.seed()                                # 初始化随机种子，增加随机性
        # 随机初始化聚类中心
        #random_index = np.random.ra
        clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters

        # 开始迭代寻求最佳聚类中心
        while True:
            # 计算目标框与聚类中心距离
            distances = 1 - self.iou(boxes, clusters)

            # 计算当前最近距离下标
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():     # 前后两次最近距离下标一样，算法收敛
                break  # clusters won't change

            # 对聚类中心（模版框）进行重新复制，
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
            # 更新前一次最近距离类别的下标
            last_nearest = current_nearest

        return clusters

    def result2txt(self, data,reuslt_path):
        '''
        这是将数据写入txt文件的韩素红
        Args:
            data: 聚类中心(模版框)
            reuslt_path: 聚类中心(模版框)结果txt文件保存路径
        Returns:
        '''
        with open(reuslt_path,'w') as f:
            row = np.shape(data)[0]
            for i in range(row):
                if i == 0:
                    x_y = "%d,%d" % (data[i][0], data[i][1])
                else:
                    x_y = ",%d,%d" % (data[i][0], data[i][1])
                f.write(x_y)

    def txt2boxes(self):
        '''
        这是从COCO格式数据集的txt文件中目标框定位信息的函数
        Returns:
        '''
        dataSet = []
        with open(self.dataset_path,"r") as f:
            for line in f.readlines():
                infos = line.split(" ")
                data = infos[1:]
                for bbox in data:
                    if ',' in bbox:
                        print(bbox.split(','))
                        xmin,ymin,xmax,ymax,id = bbox.split(',')
                        width = int(xmax) - int(xmin)
                        height = int(ymax) - int(ymin)
                        dataSet.append([width, height])
            result = np.array(dataSet)
        return result

    def txt2clusters(self,result_path):
        '''
        这是执行K-Means聚类算法，并将结果保存到txt文件的函数
        Args:
            result_path: 结果txt文件路径
        Returns:
        '''
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result,result_path)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))

def run_main():
    """
       这是主函数
    """
    cluster_number = args.cluster_number
    dataset_path = os.path.abspath(args.dataset_path)
    result_path = os.path.abspath(args.yolo_anchors_path)
    kmeans = YOLO_KMeans(cluster_number, dataset_path)
    kmeans.txt2clusters(result_path)

if __name__ == '__main__':
    run_main()