# RT-Smart 部署勘智在线训练平台模型

## 1. 概述

`cloudplat_deploy_code` 封装了勘智在线训练平台模型部署的代码，用户需要在`rtos_sdk`下编译可执行文件部署训练平台得到的模型。编译流程见：[如何编译固件](https://www.kendryte.com/k230_rtos/zh/main/userguide/how_to_build.html)

## 2. 源码说明

`cloudplat_deploy_code` 代码实现了训练平台支持的图像分类、目标检测、语义分割、OCR检测、OCR识别、双模型任务OCR检测+识别、度量学习（图像特征化）、多标签分类共计8个任务。代码封装了模型推理、预处理工具方法、配置文件解析、结果绘制的公共部分，放在`common_files`目录下，其他目录分别存放对应任务的推理代码。

### 3. 代码结构

下面是代码文件的说明：

```shell
cloudplat_deploy_code_linux
├── common_files
├── classification              #图像分类任务
├── detection                   #目标检测任务
├── segmentation                #语义分割任务
├── ocr_detection               #OCR检测任务
├── ocr_recognition             #OCR识别任务
├── ocr                         #OCR任务
├── metric_learning             #度量学习任务
├── multilabel_classification   #多标签分类任务
├── utils
│    │- SourceHanSansSC-Normal-Min.ttf  # 字体文件
│── libs                                # freetype相关的第三方库
├── CMakeLists.txt
└── build.sh
```

## 4. 编译说明

### 4.1 参数配置

您可以在`common_files`中的`setting.h`配置参数，关于参数配置的解析如下，主要用于配置屏幕显示：

|宏定义参数|说明|
|:-:|:-:|
|`ISP_WIDTH`|ISP输出宽度|
|`ISP_HEIGHT`|ISP输出高度|
|`DISPLAY_MODE`|显示模式，0为1920×1080 LT9611，1为800×480 ST7701|
|`DISPLAY_WIDTH`|显示屏幕宽度|
|`DISPLAY_HEIGHT`|显示屏幕高度|
|`AI_FRAME_WIDTH`|AI推理帧宽度|
|`AI_FRAME_HEIGHT`|AI推理帧高度|
|`AI_FRAME_CHANNEL`|AI推理帧通道数|
|`USE_OSD`|是否使用OSD，0为不使用，1为使用|
|`OSD_WIDTH`|OSD图层宽度,用于显示AI推理结果|
|`OSD_HEIGHT`|OSD图层高度,用于显示AI推理结果|
|`OSD_CHANNEL`|OSD图层通道数|

### 4.2 源码编译

进入`src/rtsmart/examples/cloudplat_deploy_code`目录

```shell
# 进入目录
cd cloudplat_deploy_code

# 编译文件,会在k230_bin目录下得到所有的任务编译elf文件
./build_app.sh

# 如果只想编译某一个任务的部署文件，可以使用./build.sh <任务名>
./build_app.sh classification
./build_app.sh detection
...

```

编译产物在 `k230_bin` 目录下。

### 4.3 上板部署

将得到的`elf文件`、字体文件和勘智训练平台得到的`kmodel`、`deploy_config.json`以及测试图片拷贝到开发板上的某一目录中，运行命令：

```shell
# 分类-视频推理，输入`q`回车退出视频推理
./classification.elf deploy_config.json None 0

# 分类-图片推理
./classification.elf deploy_config.json test.jpg 0

# 检测-视频推理，输入`q`回车退出视频推理
./detection.elf deploy_config.json None 0

# 检测-图片推理
./detection.elf deploy_config.json test.jpg 0

# 语义分割-视频推理，输入`q`回车退出视频推理
./segmentation.elf deploy_config.json None 0

# 语义分割-图片推理
./segmentation.elf deploy_config.json test.jpg 0

# OCR检测-视频推理，输入`q`回车退出视频推理
./ocr_detection.elf deploy_config.json None 0

# OCR检测-图片推理
./ocr_detection.elf deploy_config.json test.jpg 0

# OCR识别-图片推理，该任务只支持图片推理
./ocr_recognition.elf deploy_config.json test.jpg 0

# OCR-视频推理，输入`q`回车退出视频推理
./ocr.elf ocrdet_deploy_config.json ocrrec_deploy_config.json None 0

# OCR-图片推理
./ocr.elf ocrdet_deploy_config.json ocrrec_deploy_config.json test.jpg 0

# 度量学习-视频推理，输入`q`回车退出视频推理
./metric_learning.elf deploy_config.json None 0

# 度量学习-图片推理
./metric_learning.elf deploy_config.json test.jpg 0

# 多标签分类-视频推理，输入`q`回车退出视频推理
./multilabel_classification.elf deploy_config.json None 0

# 多标签分类-图片推理
./multilabel_classification.elf deploy_config.json test.jpg 0
```
