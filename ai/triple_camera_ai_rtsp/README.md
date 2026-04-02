# 三摄AI RTSP 示例

## 概述

本示例展示了在K230 RTOS平台上运行的三摄AI应用，主要功能包括：

- **摄像头0（Sensor 0）**：YOLOv8 80类目标检测 + 本地显示
- **摄像头1（Sensor 1）**：YOLOv8 80类目标检测 + 本地显示   
- **摄像头2（Sensor 2）**：YOLOv8 80类目标检测 + 本地显示 + RTSP推流

应用采用多线程并发处理三路摄像头数据流，通过互斥锁保护共享的AI资源（AI2D和KPU），支持本地屏幕显示和RTSP网络推流。

## 功能特性

- **三摄支持**：同时处理三路独立的摄像头数据流
- **目标检测**：基于YOLOv8的80类目标检测
- **本地显示**：支持MIPI屏幕（ST7701）和HDMI（LT9611）两种显示输出
- **OSD叠加**：检测框和结果实时叠加显示
- **RTSP推流**：通过RTSP协议进行实时视频推流
- **多线程处理**：并发处理三路摄像头数据

## 目录结构

```
triple_camera_ai_rtsp/
├── CMakeLists.txt          # CMake项目配置
├── Makefile                 # 编译Makefile
├── build_app.sh             # 编译脚本
├── README.md                # 本文档
├── cmake/
│   ├── Riscv64.cmake       # RISC-V交叉编译工具链
│   └── link.lds            # 链接脚本
├── src/
│   ├── main.cc             # 主入口和线程管理
│   ├── video_pipeline.cc   # 视频管线实现（VICAP、VB、VO、OSD、RTSP）
│   ├── video_pipeline.h    # 视频管线头文件
│   ├── ai_base.cc          # AI基类（nncase封装）
│   ├── ai_base.h           # AI基类头文件
│   ├── ai_utils.cc         # AI工具函数
│   ├── ai_utils.h          # AI工具函数头文件
│   ├── yolov8_detect.cc    # YOLOv8目标检测实现
│   ├── yolov8_detect.h     # YOLOv8目标检测头文件
│   ├── encoded_rtsp_server.cc # RTSP服务器实现
│   ├── encoded_rtsp_server.h  # RTSP服务器头文件
│   ├── setting.h           # 配置常量
│   └── scoped_timing.h     # 调试用计时工具
└── utils/
    ├── yolov8n_320.kmodel  # YOLOv8目标检测模型
    └── run.sh              # 运行脚本
```

## 依赖

- K230 RTOS SDK
- OpenCV（嵌入式版本）
- nncase运行时
- MPP（媒体处理平台）
- RTSP服务器库

## 编译

### 前置条件

1. 确保K230 SDK已正确配置
2. 配置RISC-V工具链：`~/.kendryte/k230_toolchains/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin`
3. 配置目标开发板：
   ```bash
   make list-def
   # 选择对应的板级配置
   ```

### 编译命令

#### 编译带应用的固件

```bash
# 配置编译选项，选择 RT-Smart UserSpace Examples Configuration --> Enable build ai examples -->  Enable Build Triple Camera AI with RTSP Programs 
make menuconfig

# 编译固件
make -j
```

编译的固件在`output/***_defconfig`目录下，烧录固件可用。

#### 仅编译应用

- 进入`src/rtsmart/examples/ai/triple_camera_ai_rtsp`，执行脚本`build_app.sh`完成编译，编译产物位于`k230_bin`目录下。

- 或者进入`src/rtsmart/examples/ai/triple_camera_ai_rtsp`，执行`make`命令完成编译，编译产物位于`rtsmart/examples/elf/ai/triple_camera_ai_rtsp/`目录下。

## 运行

如果烧录的是带应用的固件，可以取消拷贝步骤。

### 拷贝文件到开发板

将以下文件拷贝到K230开发板：
- `triple_cam_ai_rtsp.elf`
- `yolov8n_320.kmodel`
- `run.sh`

### 执行

```bash
chmod +x run.sh
./run.sh
```

或直接带参数运行：

```bash
./triple_cam_ai_rtsp.elf yolov8n_320.kmodel 0.5 0.5 0
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `yolov8_kmodel_det` | YOLOv8检测kmodel路径 | `yolov8n_320.kmodel` |
| `yolov8_conf_thres` | YOLOv8目标阈值 | `0.5` |
| `yolov8_nms_thres` | YOLOv8 NMS阈值 | `0.5` |
| `debug_mode` | 调试模式：0/1/2（不调试/简单/详细） | `0` |

### 查看RTSP推流

RTSP服务器在8554端口进行推流，注意日志打印输出：

```bash
rtsp://<开发板IP>:8554/test0
```

用户可以使用VLC拉取网络串流。

## 配置说明

[setting.h](src/setting.h)中的关键配置参数：

### 显示类型配置

通过修改`DISPLAY_TYPE`选择不同的显示输出：

| 显示类型 | 分辨率 | 说明 |
|----------|--------|------|
| `st7701` | 800x480 |四分屏显示多摄，MIPI小屏（默认） |
| `lt9611` | 1920x1080 |四分屏显示多摄，HDMI输出 |

### ST7701配置（默认）

| 常量 | 值 | 说明 |
|------|-----|------|
| `ISP_WIDTH` | 1920 | 摄像头传感器宽度 |
| `ISP_HEIGHT` | 1080 | 摄像头传感器高度 |
| `DISPLAY_MODE` | 1 | 显示模式（1=ST7701） |
| `DISPLAY_WIDTH` | 400 | 摄像头绑定屏幕通道显示宽度 |
| `DISPLAY_HEIGHT` | 240 | 摄像头绑定屏幕通道显示高度 |
| `DISPLAY_ROTATE` | 1 | 是否旋转显示 |
| `AI_FRAME_WIDTH` | 640 | AI输入帧宽度 |
| `AI_FRAME_HEIGHT` | 480 | AI输入帧高度 |
| `AI_FRAME_CHANNEL` | 3 | RGB通道数 |
| `USE_OSD` | 1 | 启用OSD叠加AI处理效果 |
| `OSD_WIDTH` | 400 | OSD宽度 |
| `OSD_HEIGHT` | 240 | OSD高度 |
| `ENABLE_RTSP_SERVER` | 1 | 启用RTSP服务 |
| `VENC_FRAME_WIDTH` | 1920 | 视频编码宽度 |
| `VENC_FRAME_HEIGHT` | 1080 | 视频编码高度 |

### LT9611配置（HDMI）

| 常量 | 值 | 说明 |
|------|-----|------|
| `ISP_WIDTH` | 1920 | 摄像头传感器宽度 |
| `ISP_HEIGHT` | 1080 | 摄像头传感器高度 |
| `DISPLAY_MODE` | 0 | 显示模式（0=LT9611） |
| `DISPLAY_WIDTH` | 400 | 摄像头绑定屏幕通道显示宽度 |
| `DISPLAY_HEIGHT` | 240 | 摄像头绑定屏幕通道显示高度 |
| `DISPLAY_ROTATE` | 0 | 不旋转 |
| `AI_FRAME_WIDTH` | 640 | AI输入帧宽度 |
| `AI_FRAME_HEIGHT` | 480 | AI输入帧高度 |
| `AI_FRAME_CHANNEL` | 3 | RGB通道数 |
| `USE_OSD` | 1 | 启用OSD叠加AI处理效果 |
| `OSD_WIDTH` | 960 | OSD宽度 |
| `OSD_HEIGHT` | 540 | OSD高度 |
| `ENABLE_RTSP_SERVER` | 1 | 启用RTSP服务 |
| `VENC_FRAME_WIDTH` | 1920 | 视频编码宽度 |
| `VENC_FRAME_HEIGHT` | 1080 | 视频编码高度 |

## AI模型说明

### YOLOv8目标检测（80类）

支持检测80类目标，包括：
- 行人、自行车、汽车、摩托车、飞机、公交车、火车、卡车、船
- 红绿灯、消防栓、停车标志、停车计时器、长椅
- 鸟、猫、狗、马、羊、牛、大象、熊、斑马、长颈鹿
- 背包、雨伞、手提包、领带、行李箱
- 滑雪板、运动球、风筝、棒球棒、滑板
- 瓶子、酒杯、杯子、叉子、刀、勺子、碗
- 香蕉、苹果、三明治、橙子、西兰花、胡萝卜、热狗、披萨、甜甜圈、蛋糕
- 椅子、沙发、盆栽、床、餐桌、马桶
- 电视、笔记本电脑、鼠标、遥控器、键盘、手机
- 微波炉、烤箱、烤面包机、水槽、冰箱
- 书、时钟、花瓶、剪刀、泰迪熊、吹风机、牙刷

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                           主线程                                  │
│                    （用户输入 'q' 退出）                           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    线程0       │     │    线程1       │     │    线程2       │
│  (Sensor 0)   │     │  (Sensor 1)   │     │  (Sensor 2)   │
│               │     │               │     │               │
│  YOLOv8检测   │     │  YOLOv8检测   │     │  YOLOv8检测   │
│  OSD叠加      │     │  OSD叠加      │     │  OSD叠加      │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    互斥锁保护    │
                    │  (AI2D + KPU)   │
                    └─────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                         ┌───────────────┐
│   本地显示     │                         │  RTSP推流     │
│  (VO + OSD)   │                         │  (VENC)       │
└───────────────┘                         └───────────────┘
```

## 数据流说明

1. **视频采集**：三路GC2093摄像头通过VICAP采集1080P视频
2. **通道分流**：处理为不同通道所需的数据格式和分辨率
   - CHN0：绑定到VO层，用于本地显示
   - CHN1：用于AI推理（dump模式获取帧）
   - CHN2：绑定到VENC，用于RTSP编码推流，仅sensor2有此功能
3. **AI处理**：将640x480的RGB图像送入YOLOv8模型进行检测
4. **OSD叠加**：检测结果绘制到OSD层，与视频叠加显示
5. **RTSP推流**：编码后的视频流通过RTSP服务推送

