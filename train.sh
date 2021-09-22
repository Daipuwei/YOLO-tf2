#!/bin/bash
#python tools/train.py --config_file_path config/BDD100k-Daytime/yolov3.yaml
#python tools/train.py --config_file_path config/BDD100k-Daytime/yolov4.yaml
python tools/train.py --config_file_path config/BDD100k-Daytime/yolov4-csp.yaml
python tools/train.py --config_file_path config/BDD100k-Daytime/yolov4-p5.yaml
python tools/train.py --config_file_path config/BDD100k-Daytime/yolov4-p6.yaml

