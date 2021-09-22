#!/bin/bash
# yolov4
python mAP/get_gt_json.py --dataset_dir ../dataset/BDD100k-Daytime --dataset_name BDD100k-Daytime --ext .jpg --classes_path ./model_data/bdd100k_classes.txt
python mAP/get_dr_json.py --dataset_dir ../dataset/BDD100k-Daytime --config_file_path ./config/BDD100k-Daytime/yolov4 --ext .jpg -h 640 -w 1280 --images_optional_flag MODEL.MODEL_PATH  ./checkpoints/BDD100k-Daytime/yolov4/640x1280/batchsize=16/use_mosaic=1/Adadelta-lr=1.0/time/train_yolo_model_stage2.h5
python mAP/compute_mAP_COCO.py --gt_json_path ./compute_mAP_COCO/input/BDD100k/gt_result.json --dr_json_path ./compute_mAP_COCO/input/BDD100k/dr_result.json
