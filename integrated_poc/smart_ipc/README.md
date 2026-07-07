# K230 综合demo - 单路智能 IPC

## 功能概述

单路智能 IPC 系统是一款集 **网络推流**、**画面渲染** 和 **人脸识别** 功能于一体的智能监控系统。该系统仅需连接一个摄像头，即可实现以下功能：

- **实时画面显示与人脸识别**：通过 HDMI 输出或 LCD 屏幕实时显示摄像头画面，并对画面中的人脸进行检测和标注。
- **网络视频推流**：将摄像头的实时画面通过网络进行推流，客户端可通过网络流地址查看实时视频。支持 **RTSP** 和 **WebRTC** 两种推流模式。
  - **RTSP 模式**：支持音视频推流，适用于 VLC 等播放器。
  - **WebRTC 模式**：支持浏览器直接观看，零插件，低延迟（< 500ms）。*注意：WebRTC 模式当前仅支持视频推流，不支持音频。*

*注意：推流的画面可包含人脸识别的标注信息，也可仅传输原始视频。*

## 使用说明

项目的源代码位于目录 `/src/rtsmart/examples/integrated_poc/smart_ipc`。

您可以通过 menuconfig 启用该示例的编译：
进入 `RT-Smart UserSpace Examples Configuration` → 勾选 `Enable build integrated examples` → 进一步启用 `Smart IPC examples`。

编译成功后，生成的可执行文件将部署至目标设备的如下路径：
`/sdcard/app/examples/integrated_poc/smart_ipc/smart_ipc.elf`

在目标设备上，可通过以下命令启动单路智能 IPC 系统：

/sdcard/app/examples/integrated_poc/smart_ipc/smart_ipc.elf [选项]

示例：

- 使用 HDMI 输出 + RTSP 推流：

    ```sh
    ./smart_ipc.elf -C 0
    ```

- 使用 LCD 输出 + RTSP 推流：

    ```sh
    ./smart_ipc.elf -C 1
    ```

- 使用 WebRTC 推流（浏览器直接观看）：

    ```sh
    ./smart_ipc.elf -M 1 -C 1
    ```

- 使用 WebRTC 推流，H.265 编码，自定义端口：

    ```sh
    ./smart_ipc.elf -M 1 -t h265 -P 9090 -C 0
    ```

查看帮助参数信息：

```sh
./smart_ipc.elf -H
```

以下是详细的命令行参数说明：

| 选项                  | 描述                                   | 默认值                      |
|-----------------------|----------------------------------------|-----------------------------|
| `-H`                  | 显示帮助信息                           | -                           |
| `-S`                  | sensor类型（默认自动探测sensor类型）    | SENSOR_TYPE_MAX             |
| `-V`                  | 是否启用音频采集（默认启用）            | 1                           |
| `-a <audio_sample>`   | 音频采样率                             | 8000                        |
| `-c <channel_count>`  | 音频通道数                             | 1                           |
| `-t <codec_type>`     | 视频编码类型，h264/h265                | h264                         |
| `-w <width>`          | 视频编码宽度                           | 1280                        |
| `-h <height>`         | 视频编码高度                           | 720                         |
| `-b <bitrate_kbps>`   | 视频编码码率(kbps)                     | 2000                        |
| `-D`                  | 编码数据源 0:从sensor通道获取原始画面编码 1：从vo wbc 获取合成画面编码 | 0 |
| `-C <connector_type>` | 视频输出连接类型 (0:HDMI,1:LCD)        | HDMI                        |
| `-A <ai_input_width>` | AI 分析输入宽度                        | 1280                        |
| `-I <ai_input_height>`| AI 分析输入高度                        | 720                         |
| `-K <kmodel_file>`    | kmodel 文件路径                        | face_detection_320.kmodel   |
| `-T <obj_thresh>`     | 人脸检测阈值                           | 0.6                         |
| `-N <nms_thresh>`     | 人脸检测 NMS 阈值                      | 0.4                         |
| `-E`                  | 是否启用画面渲染功能（0:禁用，1:启用）   | 1                           |
| `-F`                  | 是否启用ai分析功能（0:禁用，1:启用）     | 1                           |
| `-G`                  | 是否启用视频编码推流功能（0:禁用，1:启用）| 1                           |
| `-M <streaming_mode>` | 推流模式（0:RTSP，1:WebRTC）           | 0 (RTSP)                    |
| `-P <port>`           | 推流端口（RTSP 默认 8554，WebRTC 默认 8080） | 8554 (RTSP) / 8080 (WebRTC) |

## WebRTC 模式说明

### 浏览器要求

#### mDNS 隐私保护

Chrome/Edge 默认启用 mDNS 隐私保护，会隐藏本地 IP 导致 WebRTC 连接失败。页面会自动检测并提示。

修复方式（任选其一）：

1. 地址栏输入 `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns` → 设为 **Disabled** → 重启
2. 启动参数加 `--disable-features=WebRtcHideLocalIpsWithMdns`
3. 使用 Firefox（默认未启用 mDNS）

#### H.265 浏览器兼容性

| 浏览器 | H.265 WebRTC 支持 | 说明 |
|--------|-------------------|------|
| Chrome | 支持 | 需硬件 HEVC 解码支持（大部分桌面端已具备） |
| Edge | 支持 | 与 Chrome 同内核，同样依赖硬件解码 |
| Firefox | 不支持 | WebRTC 尚未实现 HEVC 解码 |
| Safari | 部分支持 | macOS/iOS 上依赖硬件解码，Windows 版不支持 |

建议：如需最大兼容性，使用默认的 H.264 编码（`-t h264`）。

### WebRTC 信令流程

```
浏览器                        开发板 (smart_ipc)
  │                               │
  │─── GET /offer ──────────────>│  创建 SDP Offer + ICE gathering
  │<── 200 OK (SDP Offer) ──────│
  │                               │
  │  pc.setRemoteDescription()   │
  │  pc.createAnswer()           │
  │                               │
  │─── POST /answer (SDP) ──────>│  设置远端 SDP
  │<── 200 OK ──────────────────│
  │                               │
  │        ICE/DTLS/SRTP 握手     │
  │<════════ 视频流 ════════════│  H.264/H.265 RTP 包
```

## 关键代码解析

项目的源代码位于目录 `/src/rtsmart/examples/integrated_poc/smart_ipc`。该目录包含以下关键模块和文件：

- `main.cpp`：入口程序，命令行参数解析，实例化并启动 `MySmartIPC`。
- `smart_ipc.cpp`/`smart_ipc.h`：智能 IPC 系统核心类 `MySmartIPC`，封装 RTSP/WebRTC 推流、信令处理、编码帧转发、人脸识别调度。
- `media.cpp`/`media.h`：媒体处理模块，封装了所有 MPI 接口，用于摄像头、显示器和音视频编解码等基础模块的初始化和启动。
- `face_detection`：人脸检测模块，实现人脸识别功能。
- `http_server.c`/`http_server.h`：HTTP 服务器（WebRTC 信令使用），单线程、支持 CORS。
- `web_page.h`：WebRTC 模式嵌入式前端页面。

`MySmartIPC` 类封装了智能摄像头系统的所有功能。在 `main` 函数中，只需实例化该类并调用 `Init` 和 `Start` 方法即可启动系统。

在 `MySmartIPC` 类内部，通过调用 `KdMedia` 中封装的 MPI 接口，完成了 VB 缓冲区的初始化，以及摄像头、显示器和音视频编解码等功能。通过初始化这些底层 MPI 接口，可以获取以下数据回调：

- 视频编码数据：`OnVEncData`
- 音频编码数据：`OnAEncData`
- AI 模块输入的视频数据：`OnAIFrameData`

推流模式通过 `StreamingMode` 枚举选择：

- **RTSP 模式**：`OnVEncData` 和 `OnAEncData` 中调用 `KdRtspServer` 实现音视频数据的网络推流。
- **WebRTC 模式**：`OnVEncData` 中调用 `peer_connection_send_video` 将 H.264/H.265 编码帧通过 WebRTC 发送到浏览器；HTTP 信令由 `OnHttpRequest` 处理（GET /offer 创建 SDP offer，POST /answer 设置远端描述）。

在 `OnAIFrameData` 方法中，调用 `FaceDetection` 实现人脸识别，并将检测结果显示在 VO 的 OSD 图层上。

## 数据流pipeline

该数据流pipeline展示了单路智能 IPC 系统的工作流程。系统启用了摄像头的三路通道，分别用于 HDMI/LCD 屏幕输出、人脸识别数据输入和编码推流。

1. **摄像头输入**：摄像头捕获实时视频数据，视频数据被分成三路进行处理。
1. **HDMI/LCD 输出**：一路视频数据通过 HDMI 或 LCD 屏幕进行实时显示。
1. **人脸识别**：另一路视频数据输入到 AI 模块进行人脸检测和识别。检测到的人脸信息会被标注在视频画面上，并显示在 HDMI/LCD 屏幕上。
1. **编码推流**：1）仅传输原始视频：对最后一路视频数据直接进行编码处理后，通过网络完成视频推流。需注意，此方案下推流的视频数据仅包含原始画面，不附带任何人脸识别标注信息 2）传输带 AI 分析结果的合成画面：先从 VO WBC（视频输出 / 合成模块）获取已叠加 AI 分析结果的合成画面，再对该合成画面进行编码处理，最终通过网络推流。此方案下，推流数据将完整携带 AI 分析产出的人脸识别等相关结果。

### RTSP 模式

![alt text](https://kendryte-download.canaan-creative.com/developer/pictures/pipeline_1.png)

### WebRTC 模式

```
VICAP (摄像头)
  ├── CHN0 ──bind──> VO (HDMI/LCD 预览)
  ├── CHN1 ──bind──> VENC (H.264/H.265) ──> peer_connection_send_video ──> 浏览器
  │                                                       ▲
  └── CHN2 ──> AI (人脸检测) ──> OSD                      │
                                                HTTP 信令 (offer/answer)
```

## 操作演示

### RTSP 模式

要在PC端通过网络获取摄像头的实时流画面，请确保开发板已连接网络并获得有效的IP地址。可以按照以下步骤检查网络配置：

1. 启动开发板后，在串口中输入 `ifconfig`，查看当前开发板的IP地址是否有效。
1. 确保开发板和PC端网络互通，在开发板上输入 `ping <PC_IP>`，检查网络连接。

如果需要通过HDMI输出画面，请将HDMI输出口连接到显示器。如果需要通过LCD屏输出画面，请将LCD屏连接到开发板。完成上述准备工作后，启动开发板并执行以下命令：

```sh
cd /sdcard/app/examples/integrated_poc/smart_ipc
./smart_ipc.elf -C 0 -D 0 # HDMI输出，rtsp传输原始画面
./smart_ipc.elf -C 1 -D 1 # LCD输出，rtsp传输带AI分析结果的合成画面
```

同时查看串口输出，会有提示"Play this stream using the URL `rtsp://<IP>:8554/test`"。在PC端使用VLC播放器打开该地址，即可通过网络接收实时画面。

![alt text](https://kendryte-download.canaan-creative.com/developer/pictures/smart_ipc.png)

### WebRTC 模式

1. 启动开发板并执行以下命令：

    ```sh
    cd /sdcard/app/examples/integrated_poc/smart_ipc
    ./smart_ipc.elf -M 1 -C 1 -t h264
    ```

1. 查看串口输出，会显示访问地址，例如：

    ```
    [WebRTC] Open in browser: http://10.100.228.193:8080
    ```

1. 在 PC 浏览器中打开该地址，点击页面上的"连接"按钮即可观看实时画面。

## 依赖

- **libpeer** — WebRTC 协议栈（含 mbedTLS、SRTP、SCTP），仅 WebRTC 模式需要
- **KdRtspServer** — RTSP 推流服务，仅 RTSP 模式需要
- K230 MPP SDK — VB、VICAP、VO、VENC、connector
