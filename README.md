# Detect-objects-Drone-Yolov5
#git clone https://github.com/ultralytics/yolov5  
#Link: https://www.analyticsvidhya.com/blog/2021/08/train-your-own-yolov5-object-detection-model/
!git clone https://github.com/shega2901/yolov5  # clone repositary from my github
#%pip install virtualenv  # creating a virtual environment - virtualenv
#!virtualenv yolov5_training_env # create the virtual environment for YOLO5 training  
Cloning into 'yolov5'...
remote: Enumerating objects: 10295, done.
remote: Counting objects: 100% (10295/10295), done.
remote: Compressing objects: 100% (2934/2934), done.
remote: Total 10295 (delta 7449), reused 10082 (delta 7335), pack-reused 0
Receiving objects: 100% (10295/10295), 65.04 MiB | 11.47 MiB/s, done.
Resolving deltas: 100% (7449/7449), done.
#from google.colab import drive
#drive.mount('/content/drive') # mount google.colab
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
#%cd /content/drive/MyDrive/Colab Notebooks/DL Tutorials/YOLO
%cd yolov5
%pip install -qr requirements.txt  # install required libraries for traning the model YOLO5
 
# The torch library different for GPU and CPU  
# Install without the torch library - remove lines from  requirements.txt with torch, torchvision:
#   torch>=1.7.0
#   torchvision>=0.8.1
# Install torch CPU version (windows):  
#!pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install torck GPU version (windows, check compatibility cuda with torch: https://pytorch.org/get-started/previous-versions/): 
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
/content/yolov5
     |████████████████████████████████| 596 kB 4.3 MB/s 
Looking in links: https://download.pytorch.org/whl/torch_stable.html
Collecting torch==1.8.0+cu111
  Downloading https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp37-cp37m-linux_x86_64.whl (1982.2 MB)
     |█████████████▌                  | 834.1 MB 1.5 MB/s eta 0:13:08tcmalloc: large alloc 1147494400 bytes == 0x560ecd5ec000 @  0x7f9fbe2cc615 0x560e93d194cc 0x560e93df947a 0x560e93d1c2ed 0x560e93e0de1d 0x560e93d8fe99 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8fd00 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93d1e039 0x560e93d61409 0x560e93d1cc52 0x560e93d8fc25 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8b915 0x560e93d1dafa 0x560e93d8bc0d 0x560e93d8a9ee
     |█████████████████               | 1055.7 MB 1.4 MB/s eta 0:11:09tcmalloc: large alloc 1434370048 bytes == 0x560f11c42000 @  0x7f9fbe2cc615 0x560e93d194cc 0x560e93df947a 0x560e93d1c2ed 0x560e93e0de1d 0x560e93d8fe99 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8fd00 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93d1e039 0x560e93d61409 0x560e93d1cc52 0x560e93d8fc25 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8b915 0x560e93d1dafa 0x560e93d8bc0d 0x560e93d8a9ee
     |█████████████████████▋          | 1336.2 MB 1.4 MB/s eta 0:07:56tcmalloc: large alloc 1792966656 bytes == 0x560e96a74000 @  0x7f9fbe2cc615 0x560e93d194cc 0x560e93df947a 0x560e93d1c2ed 0x560e93e0de1d 0x560e93d8fe99 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8fd00 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93d1e039 0x560e93d61409 0x560e93d1cc52 0x560e93d8fc25 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8b915 0x560e93d1dafa 0x560e93d8bc0d 0x560e93d8a9ee
     |███████████████████████████▎    | 1691.1 MB 1.3 MB/s eta 0:03:38tcmalloc: large alloc 2241208320 bytes == 0x560f0185c000 @  0x7f9fbe2cc615 0x560e93d194cc 0x560e93df947a 0x560e93d1c2ed 0x560e93e0de1d 0x560e93d8fe99 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8fd00 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93e0ec66 0x560e93d8bdaf 0x560e93d1e039 0x560e93d61409 0x560e93d1cc52 0x560e93d8fc25 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8b915 0x560e93d1dafa 0x560e93d8bc0d 0x560e93d8a9ee
     |████████████████████████████████| 1982.2 MB 1.4 MB/s eta 0:00:01tcmalloc: large alloc 1982251008 bytes == 0x560f871be000 @  0x7f9fbe2cb1e7 0x560e93d4f067 0x560e93d194cc 0x560e93df947a 0x560e93d1c2ed 0x560e93e0de1d 0x560e93d8fe99 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d1dafa 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee
tcmalloc: large alloc 2477817856 bytes == 0x560ffd42a000 @  0x7f9fbe2cc615 0x560e93d194cc 0x560e93df947a 0x560e93d1c2ed 0x560e93e0de1d 0x560e93d8fe99 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8bc0d 0x560e93d1dafa 0x560e93d8bc0d 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1dbda 0x560e93d8c737 0x560e93d8a9ee 0x560e93d1e271
     |████████████████████████████████| 1982.2 MB 2.5 kB/s 
Collecting torchvision==0.9.0+cu111
  Downloading https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp37-cp37m-linux_x86_64.whl (17.6 MB)
     |████████████████████████████████| 17.6 MB 1.4 MB/s 
Collecting torchaudio==0.8.0
  Downloading torchaudio-0.8.0-cp37-cp37m-manylinux1_x86_64.whl (1.9 MB)
     |████████████████████████████████| 1.9 MB 4.2 MB/s 
Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torch==1.8.0+cu111) (1.19.5)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch==1.8.0+cu111) (3.10.0.2)
Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.7/dist-packages (from torchvision==0.9.0+cu111) (7.1.2)
Installing collected packages: torch, torchvision, torchaudio
  Attempting uninstall: torch
    Found existing installation: torch 1.10.0+cu111
    Uninstalling torch-1.10.0+cu111:
      Successfully uninstalled torch-1.10.0+cu111
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.11.1+cu111
    Uninstalling torchvision-0.11.1+cu111:
      Successfully uninstalled torchvision-0.11.1+cu111
  Attempting uninstall: torchaudio
    Found existing installation: torchaudio 0.10.0+cu111
    Uninstalling torchaudio-0.10.0+cu111:
      Successfully uninstalled torchaudio-0.10.0+cu111
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchtext 0.11.0 requires torch==1.10.0, but you have torch 1.8.0+cu111 which is incompatible.
Successfully installed torch-1.8.0+cu111 torchaudio-0.8.0 torchvision-0.9.0+cu111
