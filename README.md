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
<pre><code>
<b>train:</b> weights=./weights/yolov5s.pt, cfg=, data=./data/VisDrone.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=10, batch_size=12, imgsz=640,
rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None,
bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False,
sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False,
linear_lr=False, label_smoothing=0.0, patience=100, freeze=0, save_period=-1, local_rank=-1,
entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest

<b>hyperparameters:</b> lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, 
warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, 
hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0,
mixup=0.0, copy_paste=0.0
Overriding model.yaml nc=10

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     40455  models.yolo.Detect                      [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 270 layers, 7046599 parameters, 7046599 gradients, 15.9 GFLOPs

Transferred 343/349 items from weights/yolov5s.pt
Scaled weight_decay = 0.00046875
optimizer: SGD with parameter groups 57 weight, 60 weight (no decay), 60 bias
albumentations: version 1.0.3 required by YOLOv5, but version 0.1.12 is currently installed
train: Scanning '../yolov5/VisDrone_datasets/VisDrone2019-DET-train/labels' images and labels...6471 found, 0 missing, 0 empty, 0 corrupted: 100% 6471/6471 [00:08<00:00, 777.23it/s] 
train: WARNING: ../yolov5/VisDrone_datasets/VisDrone2019-DET-train/images/0000137_02220_d_0000163.jpg: 1 duplicate labels removed
train: WARNING: ../yolov5/VisDrone_datasets/VisDrone2019-DET-train/images/0000140_00118_d_0000002.jpg: 1 duplicate labels removed
train: WARNING: ../yolov5/VisDrone_datasets/VisDrone2019-DET-train/images/9999945_00000_d_0000114.jpg: 1 duplicate labels removed
train: WARNING: ../yolov5/VisDrone_datasets/VisDrone2019-DET-train/images/9999987_00000_d_0000049.jpg: 1 duplicate labels removed
train: New cache created: ../yolov5/VisDrone_datasets/VisDrone2019-DET-train/labels.cache
val: Scanning '../yolov5/VisDrone_datasets/VisDrone2019-DET-val/labels' images and labels...548 found, 0 missing, 0 empty, 0 corrupted: 100% 548/548 [00:00<00:00, 561.50it/s]
val: New cache created: ../yolov5/VisDrone_datasets/VisDrone2019-DET-val/labels.cache
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 2.95 anchors/target, 0.933 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: WARNING: Extremely small objects found. 29644 of 343201 labels are < 3 pixels in size.
AutoAnchor: Running kmeans for 9 anchors on 342304 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.7525: 100% 1000/1000 [02:08<00:00,  7.78it/s]
AutoAnchor: thr=0.25: 0.9994 best possible recall, 5.81 anchors past thr
AutoAnchor: n=9, img_size=640, metric_all=0.367/0.752-mean/best, past_thr=0.485-mean: 3,4, 4,9, 8,6, 7,14, 15,9, 15,19, 31,17, 25,37, 55,42
AutoAnchor: New anchors saved to model. Update model *.yaml to use these anchors in the future.
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 10 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       0/9     2.55G    0.1266    0.1409   0.04928       245       640: 100% 540/540 [13:13<00:00,  1.47s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:28<00:00,  1.24s/it]
                 all        548      38759      0.379       0.15     0.0911     0.0362

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       1/9     2.86G    0.1099    0.1725   0.03813       117       640: 100% 540/540 [13:20<00:00,  1.48s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:24<00:00,  1.07s/it]
                 all        548      38759       0.35      0.187      0.123     0.0501

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       2/9     2.86G    0.1078    0.1732   0.03549       111       640: 100% 540/540 [13:20<00:00,  1.48s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:23<00:00,  1.01s/it]
                 all        548      38759      0.369      0.201      0.147     0.0632

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       3/9     2.86G    0.1047    0.1728   0.03343       178       640: 100% 540/540 [13:16<00:00,  1.48s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:24<00:00,  1.05s/it]
                 all        548      38759       0.43      0.192      0.176       0.08

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       4/9     2.86G    0.1033    0.1749   0.03211       247       640: 100% 540/540 [13:30<00:00,  1.50s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:24<00:00,  1.06s/it]
                 all        548      38759      0.439      0.202      0.189     0.0882

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       5/9     2.86G    0.1017    0.1736    0.0313       267       640: 100% 540/540 [13:25<00:00,  1.49s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:23<00:00,  1.00s/it]
                 all        548      38759      0.413      0.231      0.211      0.102

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       6/9     2.86G    0.1005    0.1716   0.03046       264       640: 100% 540/540 [13:26<00:00,  1.49s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:21<00:00,  1.09it/s]
                 all        548      38759      0.422       0.23      0.215      0.103

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       7/9     2.86G   0.09936    0.1701   0.02983       517       640: 100% 540/540 [13:21<00:00,  1.48s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:21<00:00,  1.06it/s]
                 all        548      38759      0.301      0.255      0.227      0.112

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       8/9     2.86G    0.0991    0.1684   0.02934       176       640: 100% 540/540 [13:19<00:00,  1.48s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:21<00:00,  1.07it/s]
                 all        548      38759      0.335      0.275      0.239       0.12

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       9/9     2.86G    0.0982    0.1654   0.02902       345       640: 100% 540/540 [13:23<00:00,  1.49s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:20<00:00,  1.12it/s]
                 all        548      38759      0.327      0.278      0.244      0.123

<b>10 epochs completed in 2.295 hours.</b>
Optimizer stripped from <b>runs/train/exp/weights/last.pt</b>, 14.4MB
Optimizer stripped from <b>runs/train/exp/weights/best.pt</b>, 14.4MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 213 layers, 7037095 parameters, 0 gradients, 15.9 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 23/23 [00:44<00:00,  1.94s/it]
                 all        548      38759      0.318      0.283      0.244      0.123
          pedestrian        548       8844       0.33      0.404      0.345      0.134
              people        548       5125      0.345      0.326      0.262     0.0804
             bicycle        548       1287      0.135     0.0437     0.0403     0.0129
                 car        548      14064      0.445      0.734      0.686      0.434
                 van        548       1975      0.303      0.253       0.21      0.138
               truck        548        750       0.29      0.252      0.196      0.113
            tricycle        548       1045      0.328      0.044     0.0876     0.0465
     awning-tricycle        548        532      0.228      0.015     0.0354     0.0202
                 bus        548        251      0.406      0.332      0.247      0.142
               motor        548       4886      0.368      0.423      0.326      0.107
<b>Results saved to runs/train/exp</b>
</code></pre><br>
</details>

<details>
<summary>Validate model YOLO on VisDrone2019-val dataset</summary>
Checking model accuracy on VisDrone2019 val datasets. Using the weights file - best.pt
<pre><code> python val.py --weights runs/train/exp/weights/best.pt --data data/VisDrone.yaml --img 640 --iou 0.65 --half</code></pre><br>
<b>Command explanation:</b><br>
<b>val.py:</b> python file containing the training code.<br>
<b>img:</b> image size defaulted to 640<br>
<b>data:</b> the path of your YAML file.<br>
 <b>task:</b> = val, default = val<br>
<b>weights:</b> The path to the weights file created during training.<br>
<pre><code>  
val: data=data/VisDrone.yaml, weights=['runs/train/exp/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.65, task=val, 
device=, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, 
save_json=False, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False

Fusing layers... 
Model Summary: 213 layers, 7037095 parameters, 0 gradients, 15.9 GFLOPs
val: Scanning '../yolov5/VisDrone_datasets/VisDrone2019-DET-val/labels.cache' images and labels... 548 found, 
0 missing, 0 empty, 0 corrupted: 100% 548/548 [00:00<?, ?it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 18/18 [00:44<00:00,  2.45s/it]
                 all        548      38759       0.31       0.28      0.239      0.121
          pedestrian        548       8844      0.307      0.404      0.336      0.132
              people        548       5125      0.321      0.328      0.251     0.0772
             bicycle        548       1287      0.148     0.0396     0.0388     0.0127
                 car        548      14064      0.413      0.737       0.68       0.43
                 van        548       1975      0.312      0.244      0.209      0.137
               truck        548        750      0.286      0.247      0.194      0.112
            tricycle        548       1045      0.347     0.0421     0.0855     0.0463
     awning-tricycle        548        532      0.216     0.0132     0.0342     0.0198
                 bus        548        251      0.404      0.331       0.25      0.144
               motor        548       4886      0.345       0.42      0.315      0.104
Speed: 0.2ms pre-process, 12.3ms inference, 20.2ms NMS per image at shape (32, 3, 640, 640)
<b>Results saved to runs/val/exp</b>
</code></pre><br>
</details>

<details>
<summary>Validate model YOLO on VisDrone2019-test-dev dataset</summary>
Checking model accuracy on VisDrone2019 test dev datasets. Using the weights file - best.pt
<pre><code> python val.py --weights runs/train/exp/weights/best.pt --task test --data data/VisDrone.yaml --img 640 --iou 0.65 --half</code></pre><br>
<b>Command explanation:</b><br>
<b>val.py:</b> python file containing the training code.<br>
<b>img:</b> image size defaulted to 640<br>
<b>data:</b> the path of your YAML file.<br>
<b>task:</b> = test<br>
<b>weights:</b> The path to the weights file created during training.<br>
<pre><code>  
val: data=data/VisDrone.yaml, weights=['runs/train/exp/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.65,
task=test, device=, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, 
save_json=False, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False

Fusing layers... 
Model Summary: 213 layers, 7037095 parameters, 0 gradients, 15.9 GFLOPs
test: Scanning '../yolov5/VisDrone_datasets/VisDrone2019-DET-test-dev/labels' images and labels...1610 found, 0 missing, 0 empty, 0 corrupted: 100% 1610/1610 [00:03<00:00, 499.95it/s]
test: New cache created: ../yolov5/VisDrone_datasets/VisDrone2019-DET-test-dev/labels.cache
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 51/51 [01:24<00:00,  1.65s/it]
                 all       1610      75102      0.323      0.249      0.214      0.108
          pedestrian       1610      21006       0.33      0.243      0.214     0.0762
              people       1610       6376      0.334      0.144      0.126     0.0348
             bicycle       1610       1302      0.315       0.02     0.0592     0.0202
                 car       1610      28074      0.425      0.722      0.649      0.373
                 van       1610       5771      0.248      0.283      0.192      0.114
               truck       1610       2659      0.218      0.334      0.182     0.0947
            tricycle       1610        530       0.22     0.0264     0.0383     0.0202
     awning-tricycle       1610        599      0.291    0.00755     0.0307     0.0137
                 bus       1610       2940       0.51      0.477      0.464      0.278
               motor       1610       5845      0.341      0.237      0.182     0.0592
Speed: 0.2ms pre-process, 12.9ms inference, 15.5ms NMS per image at shape (32, 3, 640, 640)
<b>Results saved to runs/val/exp2</b>
</code></pre><br>
</details>
