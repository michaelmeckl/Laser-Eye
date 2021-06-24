Adjusted the code (Original author: 1996scarlet) to fit our needs for the "Praxisseminar" course at 
the University of Regensburg in the summer semester 2021 as a webcam eye-tracking
library that would work well enough to extract precise enough data to measure a
user's workload while playing a video game.

### Requirements:
Python 3.9 is recommended, but 3.8 and 3.7 should also work fine.

For other requirements have a look at the [requirements.txt](requirements.txt).

### Generating an exe file for the tracking system:
1. Comment out the config parsing in the logger and add the server credentials directly in the code.

2. For creating exe with auto-py-to-exe:
* select --onefile and window mode
* add-data: ```C:/Users/[user]/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/weights```
* add-data: ```C:/Users/[user]/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet``` (or wherever mxnet 
  has been installed)
* add-data: ```C:/Users/[user]/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking_service```
* add hidden-imports: ```pandas```, ```pysftp```, ```requests``` and ```plyer.platforms.win.notification```
  (for plyer import see https://stackoverflow.com/questions/56281839/issue-with-plyer-library-of-python-when-creating-a-executable-using-pyinstaller)

3. Pyinstaller command for the above:
```shell
pyinstaller --noconfirm --onefile --windowed --add-data "C:/Users/[user]/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet;mxnet/" --add-data "C:/Users/[user]/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/weights;weights/" --add-data "C:/Users/[user]/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking_service;tracking_service/" --hidden-import "plyer.platforms.win.notification" --hidden-import "pandas" --hidden-import "pysftp" --hidden-import "requests"  "C:/Users/[user]/Documents/GitHub/Praxisseminar-Webcam-Tracking-System/tracking/tracker.py"
```

# Original Documentation below:

## Laser Eye : Gaze Estimation via Deep Neural Networks

![BootJump](./asset/logo.webp)

## Installation

### Requirements

* Python 3.5+
* Linux, Windows or macOS
* mxnet (>=1.4)

While not required, for optimal performance(especially for the detector) it is highly recommended to run the code using a CUDA enabled GPU.

### Run

* Prepare an usb camera or some videos
* [Optional] More Accurate Face Alignment
  * Download [Hourglass2(d=3)-CAB pre-trained model](https://github.com/deepinx/deep-face-alignment)
  * Replace `MobileAlignmentorModel` with `CabAlignmentorModel`
* `python3 video_gaze_test.py`

### Tips

* Edit`MxnetDetectionModel`'s `scale` parameter to make a suitable input size of face detector
* More details at [Wiki](https://github.com/1996scarlet/Laser-Eye/wiki)

## Gaze Estimation for MMD Face Capture

<p align="center"><img src="https://s1.ax1x.com/2020/10/24/BVmyWt.gif" /></p>

* Try with [Open Vtuber](https://github.com/1996scarlet/OpenVtuber)

## Face Detection

* [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
* [faster-mobile-retinaface (MobileNet Backbone)](https://github.com/1996scarlet/faster-mobile-retinaface)

## Facial Landmarks Detection

* MobileNet-v2 version (1.4MB, using by default)
* [Hourglass2(d=3)-CAB version (37MB)](https://github.com/deepinx/deep-face-alignment)

## Head Pose Estimation

* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)

## Iris Segmentation

* [U-Net version (model size 71KB, TVCG 2019)](https://ieeexplore.ieee.org/document/8818661)

## Citation

``` bibtex
@article{wang2019realtime,
  title={Realtime and Accurate 3D Eye Gaze Capture with DCNN-based Iris and Pupil Segmentation},
  author={Wang, Zhiyong and Chai, Jinxiang and Xia, Shihong},
  journal={IEEE transactions on visualization and computer graphics},
  year={2019},
  publisher={IEEE}
}

@inproceedings{deng2019retinaface,
  title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
  booktitle={arxiv},
  year={2019}
}
```
