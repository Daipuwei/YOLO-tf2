#!/bin/bash
python tools/kmeans.py --cluster_number 9 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov3_anchors.txt
python tools/kmeans.py --cluster_number 9 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov3_spp_anchors.txt
python tools/kmeans.py --cluster_number 9 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov4_anchors.txt
python tools/kmeans.py --cluster_number 9 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov4_csp_anchors.txt
python tools/kmeans.py --cluster_number 12 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov4_p5_anchors.txt
python tools/kmeans.py --cluster_number 16 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov4_p6_anchors.txt
python tools/kmeans.py --cluster_number 20 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov4_p7_anchors.txt
python tools/kmeans.py --cluster_number 6 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov3_tiny_anchors.txt
python tools/kmeans.py --cluster_number 6 --dataset_path model_data/COCO/BDD100k-Daytime/trainval.txt --yolo_anchors_path model_data/bdd100k_yolov4_tiny_anchors.txt

