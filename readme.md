# ESE Capstone Design project

Embedded System Engineering Capstone Design Project Repository



## Members

- 강병국, 김세진, 나요한, 이광우



## Requirements

- Python == 3.8.11 (3.9 Not worked!)
- MediaPipe == 0.8.2
- OpenCV == 4.5.3
- numpy == 1.19.3
- sklearn == 1.0
- pandas == 1.3.3

```shel
conda create -n home-training python=3.8
conda activate home-training

pip install mediapipe==0.8.2
pip install sklearn
pip install pandas
```

```
pip install -U albumentations
pip install torchmetrics
pip install tqdm
pip install timm
```
  

## Folder Structure

```
home-training/
│
├── AiTrainerProject.py - main script to start training
├── findProject.py - 
│
├── poseestimationmin.py - holds configuration for training
├── PoseModule.py - class to handle config file and cli options
│
├── handtracking/ - abstract base classes
    ├── Handtrackingmodule.py
    ├── main.py
    └── myNewname_handtracking.py

```



## Usage

```
# upload videos "1.mp4"

python3 AitrainerProject.py
```



## Install Issue

Original error was: DLL load failed while importing _multiarray_umath:

- Python 3.9 version matches numpy 1.21. but opencv 4.5.3 can't  compatible with numpy 1.21.
- Install Python 3.8 and numpy 1.19



Backend TkAgg is interactive backend. Turning interactive mode on.

- check mediapipe version is upper than 0.8.2

- reinstall mediapipe==0.8.2 than automatically install opencv==4.5.3, numpy==1.19.3
