#!/bin/bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# VisDrone2019-DET dataset https://github.com/VisDrone/VisDrone-Dataset 
# Example usage: bash data/scripts/get_visdrone.sh
# parent
# ├── yolov5
# └── VisDrone_datasets
#     └──   ← downloads here

# Download/unzip visdrone 
python - <<EOF
from utils.general import download

dir = 'VisDrone_datasets'  # dataset root dir
urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
download(urls, dir=dir) # 
EOF
