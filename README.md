## <div align="center">Detect-objects-Drone-Yolov5</div>

The model used was YOLO. Used ["Colab"](https://colab.research.google.com) was used. The VisDrone 2019 dataset was used. This project was built using the repository [ultralytics / yolov5](https://github.com/ultralytics/yolov5) 

## <div align="center">Start</div>
To begin, upload Project_YOLO5.ipynb to the Collaboratory:
1. Visit [the Colaboratory page](https://colab.research.google.com/) in a new tab
2. From the menu "File," open the notebook
3. Then select GitHub. Enter a GitHub URL: shega2901. Reposytory: shega2901/Detect-objects-Drone-Yolov5. Branch: master.<br>
    Then upload Project_YOLO5.ipynb from Path: 
    ![Open Project](/PictureReadme/Colab1.jpg)
4. Mount Google Drive
5. From the menu "File" save a copy in Drive
## <div align="center">An explanation of the Project_YOLOv5.ipynb algorithm</div>


<details open>
<summary>Clone&Install</summary>
Cloning repositary to colab disk. Rename folder "Detect-objects-Drone-Yolov5" to "yolov5". Install required libraries for traning the model<br>
 
```bash
$ git clone https://github.com/shega2901/Detect-objects-Drone-Yolov5
$ %mv Detect-objects-Drone-Yolov5 yolov5  
$ cd yolov5
$ pip install -qr requirements.txt
```
</details>
<br>
<details open>
<summary>Preparing to train the model YOLO</summary>
 <b>1. Pre-weights for model YOLO</b><br>
 Pre-weights downloading from https://github.com/ultralytics/yolov5/releases/download<br>
 Downloading pre-weights to folder <b>yolov5/weights</b> for training model yolov5.<br>
 Using script download_weights.sh: https://github.com/shega2901/Detect-objects-Drone-Yolov5/data/scripts/download_weights.sh <br>
<pre><code> bash data/scripts/download_weights.sh</code></pre>
<pre><code>
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt to yolov5n.pt...
100% 3.77M/3.77M [00:01<00:00, 2.20MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt to yolov5s.pt...
100% 14.0M/14.0M [00:02<00:00, 6.59MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt to yolov5m.pt...
100% 40.7M/40.7M [00:03<00:00, 11.0MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt to yolov5l.pt...
100% 89.2M/89.2M [00:04<00:00, 20.0MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt to yolov5x.pt...
100% 166M/166M [00:10<00:00, 16.9MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt to yolov5n6.pt...
100% 6.56M/6.56M [00:01<00:00, 3.77MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s6.pt to yolov5s6.pt...
100% 24.5M/24.5M [00:02<00:00, 8.67MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m6.pt to yolov5m6.pt...
100% 68.7M/68.7M [00:05<00:00, 13.2MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l6.pt to yolov5l6.pt...
100% 147M/147M [00:07<00:00, 19.5MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x6.pt to yolov5x6.pt...
100% 269M/269M [00:16<00:00, 17.6MB/s]
</code></pre><br><br>
<b>2. Download dataset VISDrone2019 for train,val,test </b><br>
Download dataset VISDrone2019 for train,val,test from https://github.com/ultralytics/yolov5/releases/download to folder <b>VisDrone_datasets</b><br>
Using script download_weights.sh:https://github.com/shega2901/Detect-objects-Drone-Yolov5/data/scripts/get_visdrone.sh <br>
<pre><code>!bash data/scripts/get_visdrone.sh</code></pre>
<pre><code>  
Downloading https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip to VisDrone_datasets/VisDrone2019-DET-train.zip...
100% 1.44G/1.44G [02:00<00:00, 12.9MB/s]
Unzipping VisDrone_datasets/VisDrone2019-DET-train.zip...
Downloading https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip to VisDrone_datasets/VisDrone2019-DET-val.zip...
100% 77.9M/77.9M [00:04<00:00, 20.0MB/s]
Unzipping VisDrone_datasets/VisDrone2019-DET-val.zip...
Downloading https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip to VisDrone_datasets/VisDrone2019-DET-test-dev.zip...
100% 297M/297M [01:13<00:00, 4.23MB/s]
Unzipping VisDrone_datasets/VisDrone2019-DET-test-dev.zip...
Downloading https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip to VisDrone_datasets/VisDrone2019-DET-test-challenge.zip...
100% 292M/292M [01:04<00:00, 4.77MB/s]
Unzipping VisDrone_datasets/VisDrone2019-DET-test-challenge.zip... 
</code></pre><br>
 
 
<b>3. Convert VisDrone dataset to YOLO dataset</b><br>
For each picture in the "images" folder, there is an "annotations" folder containing a text file.<br>
Each file from the Annotation package stores the detection results for the corresponding image, <br>
with each line containing an instance of an object in the image. The format of each line is as follows:  
`<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>`  
These text files need to generate labels for each image in YOLO format:<br>
`<class> <x_center> <y_center> <width> <height>`<br>
where:<br>
`<class> = <object_category> if <score> = 1`<br>
`<x_center> = (<bbox_left>+<bbox_width>)/2`<br>
`<y_center> = (<bbox_top>+<bbox_height>)/2`<br>
`<width> = <bbox_width>`<br>
`<height> = <bbox_height>`<br>
 Using script [convert_VisDrone_to_Yolo.sh](/data/scripts/convert_VisDrone_to_Yolo.sh) <br>
`!bash data/scripts/convert_VisDrone_to_Yolo.sh`<br>
<pre><code>VisDrone_datasets/VisDrone2019-DET-train
100% 6471/6471 [01:00<00:00, 107.22it/s]
VisDrone_datasets/VisDrone2019-DET-val
100% 548/548 [00:06<00:00, 79.72it/s]
VisDrone_datasets/VisDrone2019-DET-test-dev
100% 1610/1610 [00:13<00:00, 119.04it/s]
</code></pre><br>
 As a result of executing the script, a labels folder is created in each folder from the VisDrone2019 dataset<br>

<details>
<summary>Train model YOLO on VisDrone2019 dataset</summary>
<pre><code> python train.py --img 640 --batch 12 --epochs 10 --data ./data/VisDrone.yaml --weights ./weights/yolov5s.pt</code></pre><br>
<b>Command explanation:</b><br>
<b>train.py:</b> python file containing the training code.<br>
<b>img:</b> image size defaulted to 640<br>
<b>batch:</b> batch size which is again directly dependent on your memory.<br>
<b>data:</b> the path of your YAML file.<br>
<b>weights:</b> the path of pre-weights file that has downloaded.<br>
<b>epochs:</b> number of passes of the entire training the neural network with all the training data<br>
Once training is completed in the "YoloV5/runs/train" folder are two weights files, <b>"best.pt"</b> and <b>"last.pt"</b> which are the trained weights.<br>
</details>




`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            img.jpg  # image
                            vid.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results
on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on
first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the
largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Roboflow for Datasets, Labeling, and Active Learning](https://github.com/ultralytics/yolov5/issues/4975)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

</details>

## <div align="center">Environments</div>

Get started in seconds with our verified environments. Click each icon below for details.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
</div>

## <div align="center">Integrations</div>

<div align="center">
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-long.png" width="49%"/>
    </a>
    <a href="https://roboflow.com/?ref=ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-roboflow-long.png" width="49%"/>
    </a>
</div>

|Weights and Biases|Roboflow ⭐ NEW|
|:-:|:-:|
|Automatically track and visualize all your YOLOv5 training runs in the cloud with [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme)|Label and export your custom datasets directly to YOLOv5 for training with [Roboflow](https://roboflow.com/?ref=ultralytics) |


<!-- ## <div align="center">Compete and Win</div>

We are super excited about our first-ever Ultralytics YOLOv5 🚀 EXPORT Competition with **$10,000** in cash prizes!

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/discussions/3213">
  <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"></a>
</p> -->

## <div align="center">Why YOLOv5</div>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/136901921-abcfcd9d-f978-4942-9b97-0e3f202907df.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/136763877-b174052b-c12f-48d2-8bc4-545e3853398e.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>

* **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
* **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
