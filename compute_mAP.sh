#!/bin/bash
# yolov4
python mAP/get_gt_txt.py --dataset_dir ../dataset/BDD100k-Daytime
python mAP/get_dr_txt.py --dataset_dir ../dataset/BDD100k-Daytime --config_file_path ./config/BDD100k-Daytime/yolov4 --ext .jpg -h 640 -w 1280 --images_optional_flag MODEL.MODEL_PATH  ./checkpoints/BDD100k-Daytime/yolov4/640x1280/batchsize=16/use_mosaic=1/Adadelta-lr=1.0/time/train_yolo_model_stage2.h5
python mAP/compute_mAP.py --MINOVERLAP 0.5 --results_files_path ./mAP/results/BDD100k-Daytime/yolov4/640x1280
