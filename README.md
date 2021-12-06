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

<details open><summary>Clone&Install</summary>
Cloning repositary to colab disk. Rename folder "Detect-objects-Drone-Yolov5" to "yolov5". Install required libraries for traning the model<br>
 
```bash
$ git clone https://github.com/shega2901/Detect-objects-Drone-Yolov5
$ %mv Detect-objects-Drone-Yolov5 yolov5  
$ cd yolov5
$ pip install -qr requirements.txt
```
</details>
<br>
<details open><summary>Preparing to train the model YOLO</summary>
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
</details>
<details open><summary>Train model YOLO on VisDrone2019 dataset</summary>
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
<br>
<b>Results saved to runs/train/exp</b>
</code></pre><br>
    <b>Results Graphs of traning model YOLOv5</b><br>
    Graph of epochs=10<br>
    ![epochs](/runs/train/exp/results.png)<br>
    Graph F1<br>
    ![epochs](/runs/train/exp/F1_curve.png)<br>
    Graph P_curve<br>
    ![epochs](/runs/train/exp/P_curve.png)<br>
    Graph R_curve<br>
    ![epochs](/runs/train/exp/R_curve.png)<br>
    Graph PR_curve<br>
    ![epochs](/runs/train/exp/PR_curve.png)<br>
</details>
    
    
<details open><summary><b>Validate model YOLO on VisDrone2019-val dataset</b></summary>
Checking model accuracy on VisDrone2019 val datasets. Using the weights file - best.pt
<pre><code> python val.py --weights runs/train/exp/weights/best.pt --data data/VisDrone.yaml --img 640 --iou 0.65 --half</code></pre><br>
<b>Command explanation:</b><br>
<b>val.py:</b> python file containing the training code.<br>
<b>img:</b> image size defaulted to 640<br>
<b>data:</b> the path of VisDrone.yaml file.<br>
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
<b>Results Graphs of validate model YOLOv5</b><br>
    Graph F1<br>
    ![epochs](/runs/val/exp/F1_curve.png)<br>
    Graph P_curve<br>
    ![epochs](/runs/val/exp/P_curve.png)<br>
    Graph R_curve<br>
    ![epochs](/runs/val/exp/R_curve.png)<br>
    Graph PR_curve<br>
    ![epochs](/runs/val/exp/PR_curve.png)<br>
</details>
<details open><summary>Validate-testing model YOLO on VisDrone2019-test-dev dataset</summary>
Checking model accuracy on VisDrone2019 test dev datasets. Using the weights file - best.pt
<pre><code> python val.py --weights runs/train/exp/weights/best.pt --task test --data data/VisDrone.yaml --img 640 --iou 0.65 --half</code></pre><br>
<b>Command explanation:</b><br>
<b>val.py:</b> python file containing the training code.<br>
<b>img:</b> image size defaulted to 640<br>
<b>data:</b> the path of VisDrone.yaml file.<br>
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
<b>Results saved to runs/test/exp</b>
</code></pre><br>
<b>Results Graphs of Validate-testing model YOLOv5</b><br>
    Graph F1<br>
    ![epochs](/runs/test/exp/F1_curve.png)<br>
    Graph P_curve<br>
    ![epochs](/runs/test/exp/P_curve.png)<br>
    Graph R_curve<br>
    ![epochs](/runs/test/exp/R_curve.png)<br>
    Graph PR_curve<br>
    ![epochs](/runs/test/exp/PR_curve.png)<br>
</details>


<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the latest YOLOv5 release and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            img.jpg  # image
                            vid.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

Checking model accuracy on video. Using the weights file - best.pt
<pre><code> python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source data/video</code></pre><br>
<b>Command explanation:</b><br>
<b>detect.py:</b> python file containing the detect code.<br>
<b>img:</b> image size defaulted to 640<br>
<b>source:</b> type of source.<br>
<b>task:</b> = test<br>
<b>weights:</b> The path to the weights file created during training.<br>
<pre><code>
detect: weights=['runs/train/exp/weights/best.pt'], source=data/video, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, 
device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, 
agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, 
line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False

Fusing layers... 
Model Summary: 213 layers, 7037095 parameters, 0 gradients, 15.9 GFLOPs
video 1/1 (1/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.030s)
video 1/1 (2/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (3/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (4/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (5/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (6/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (7/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (8/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (9/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (10/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (11/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (12/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (13/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (14/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (15/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (16/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (17/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (18/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (19/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 3 pedestrians, 11 cars, Done. (0.028s)
video 1/1 (20/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (21/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (22/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 1 pedestrian, 11 cars, Done. (0.028s)
video 1/1 (23/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 1 pedestrian, 11 cars, Done. (0.028s)
video 1/1 (24/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 1 pedestrian, 11 cars, Done. (0.028s)
video 1/1 (25/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (26/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (27/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (28/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (29/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (30/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (31/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (32/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (33/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (34/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, 1 van, Done. (0.028s)
video 1/1 (35/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (36/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (37/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (38/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (39/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (40/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (41/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (42/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (43/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (44/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (45/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (46/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (47/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (48/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, 1 van, Done. (0.028s)
video 1/1 (49/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (50/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (51/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (52/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (53/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, 1 van, Done. (0.028s)
video 1/1 (54/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (55/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (56/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, 1 van, Done. (0.028s)
video 1/1 (57/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (58/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (59/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (60/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (61/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (62/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (63/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (64/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (65/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (66/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (67/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (68/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (69/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (70/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (71/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (72/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (73/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (74/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (75/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (76/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (77/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (78/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (79/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (80/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (81/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.030s)
video 1/1 (82/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (83/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.027s)
video 1/1 (84/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (85/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (86/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (87/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (88/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (89/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (90/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (91/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (92/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (93/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (94/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 13 cars, Done. (0.028s)
video 1/1 (95/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (96/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 13 cars, Done. (0.028s)
video 1/1 (97/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 14 cars, Done. (0.028s)
video 1/1 (98/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (99/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 13 cars, Done. (0.028s)
video 1/1 (100/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (101/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, 1 van, 1 truck, Done. (0.028s)
video 1/1 (102/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, 1 van, Done. (0.028s)
video 1/1 (103/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, 1 van, Done. (0.028s)
video 1/1 (104/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, 1 van, Done. (0.028s)
video 1/1 (105/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, 1 van, Done. (0.028s)
video 1/1 (106/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, 1 van, Done. (0.028s)
video 1/1 (107/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, 1 van, Done. (0.030s)
video 1/1 (108/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, 1 van, Done. (0.028s)
video 1/1 (109/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (110/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (111/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 13 cars, Done. (0.028s)
video 1/1 (112/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 11 cars, Done. (0.028s)
video 1/1 (113/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 12 cars, Done. (0.028s)
video 1/1 (114/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (115/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (116/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 10 cars, Done. (0.028s)
video 1/1 (117/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (118/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (119/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (120/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (121/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (122/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (123/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (124/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (125/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (126/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (127/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (128/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (129/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (130/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (131/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (132/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (133/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (134/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (135/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (136/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (137/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (138/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (139/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (140/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (141/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (142/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (143/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (144/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (145/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (146/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (147/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (148/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (149/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (150/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (151/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (152/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (153/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (154/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (155/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (156/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (157/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (158/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (159/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (160/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (161/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (162/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (163/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (164/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (165/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (166/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (167/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (168/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (169/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (170/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (171/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (172/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (173/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (174/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (175/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (176/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (177/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (178/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (179/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (180/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (181/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (182/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (183/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (184/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (185/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (186/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (187/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (188/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (189/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (190/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (191/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (192/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (193/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (194/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (195/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (196/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (197/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (198/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (199/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (200/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.029s)
video 1/1 (201/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (202/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (203/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (204/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (205/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (206/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 1 pedestrian, 7 cars, Done. (0.028s)
video 1/1 (207/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (208/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 1 pedestrian, 7 cars, Done. (0.028s)
video 1/1 (209/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (210/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (211/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.027s)
video 1/1 (212/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (213/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (214/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (215/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (216/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (217/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (218/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (219/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (220/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (221/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (222/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (223/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (224/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (225/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (226/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (227/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (228/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (229/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (230/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (231/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (232/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.027s)
video 1/1 (233/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (234/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (235/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (236/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (237/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (238/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (239/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (240/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (241/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (242/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (243/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (244/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (245/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (246/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (247/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (248/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (249/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (250/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (251/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (252/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (253/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (254/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (255/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (256/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (257/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (258/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (259/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (260/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (261/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (262/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (263/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (264/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (265/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (266/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (267/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (268/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (269/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (270/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (271/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (272/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (273/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (274/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (275/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (276/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (277/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (278/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (279/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (280/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (281/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (282/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (283/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (284/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (285/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (286/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (287/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (288/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (289/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (290/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (291/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (292/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (293/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (294/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (295/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (296/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (297/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (298/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (299/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (300/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (301/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (302/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (303/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (304/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (305/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (306/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 9 cars, Done. (0.028s)
video 1/1 (307/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (308/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (309/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 8 cars, Done. (0.028s)
video 1/1 (310/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (311/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (312/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.029s)
video 1/1 (313/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 7 cars, Done. (0.028s)
video 1/1 (314/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (315/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (316/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (317/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (318/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (319/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (320/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (321/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (322/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (323/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (324/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (325/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (326/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (327/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (328/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (329/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (330/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (331/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (332/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (333/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (334/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (335/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (336/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (337/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (338/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (339/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (340/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (341/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (342/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (343/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (344/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (345/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (346/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (347/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (348/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (349/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (350/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (351/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (352/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (353/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (354/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (355/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (356/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (357/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (358/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (359/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (360/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (361/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (362/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (363/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (364/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (365/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (366/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (367/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (368/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (369/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (370/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (371/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (372/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (373/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (374/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (375/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (376/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (377/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (378/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (379/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (380/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (381/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (382/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.029s)
video 1/1 (383/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (384/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (385/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (386/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (387/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (388/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (389/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (390/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.027s)
video 1/1 (391/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (392/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (393/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (394/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (395/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.027s)
video 1/1 (396/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (397/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (398/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 4 cars, Done. (0.028s)
video 1/1 (399/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 4 cars, Done. (0.027s)
video 1/1 (400/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 4 cars, Done. (0.027s)
video 1/1 (401/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 4 cars, Done. (0.028s)
video 1/1 (402/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (403/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (404/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (405/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (406/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (407/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (408/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (409/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (410/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (411/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.027s)
video 1/1 (412/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (413/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (414/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (415/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.027s)
video 1/1 (416/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (417/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (418/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (419/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (420/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (421/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (422/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (423/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (424/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (425/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (426/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (427/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (428/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (429/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.027s)
video 1/1 (430/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (431/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (432/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 5 cars, Done. (0.028s)
video 1/1 (433/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (434/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (435/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
video 1/1 (436/436) /content/yolov5/data/video/181015_01_011.mp4: 384x640 6 cars, Done. (0.028s)
Speed: 0.9ms pre-process, 27.8ms inference, 1.6ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp
</code></pre>
Input video<br>
    ![videoI](/data/video/181015-01-011_640.gif)<br>
Output video<br>
    ![videoO](/runs/detect/exp/181015-01-011_640.gif)
</details>


