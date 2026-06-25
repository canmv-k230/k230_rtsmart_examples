# sample_webrtc

K230 RT-Smart 上的 WebRTC 局域网摄像头 Demo。板载摄像头采集 → 硬件 H.264 编码 → WebRTC 推流到浏览器，零拷贝。

## 功能

- H.264 视频实时推流（CBR，分辨率/码率可配）
- LCD / HDMI 本地预览
- 浏览器一键连接，自带 Web UI
- SDP offer/answer 信令（HTTP）
- mDNS 隐私检测与提示

## 快速开始

```bash
# 1. menuconfig 启用 libpeer 和 sample_webrtc
#    RT-Smart UserSpace Libraries Configuration  →  Enable Build libpeer (WebRTC)  = Y
#    RT-Smart UserSpace Examples Configuration   →  Enable MPP examples           = Y
#                                                   →  Enable Build sample_webrtc   = Y
make menuconfig

# 2. 编译（在 K230 RT-Smart SDK 根目录）
make

# 3. 运行
sample_webrtc.elf -p 8080 -s 2 -c 605274512 -W 1280 -H 720 -b 2000

# 4. 浏览器访问
# http://<设备IP>:8080
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p` | HTTP 端口 | `8080` |
| `-s` | CSI 设备号 (0-2) | `2` |
| `-c` | 显示接口：`605274512`=LCD, `757006876`=HDMI | LCD |
| `-W` | 编码宽度 | `1280` |
| `-H` | 编码高度 | `720` |
| `-b` | 编码码率 (kbps) | `2000` |

## 浏览器要求

Chrome/Edge 默认启用 mDNS 隐私保护，会隐藏本地 IP 导致 WebRTC 连接失败。页面会自动检测并提示。

修复方式（任选其一）：

1. 地址栏输入 `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns` → 设为 **Disabled** → 重启
2. 启动参数加 `--disable-features=WebRtcHideLocalIpsWithMdns`
3. 使用 Firefox（默认未启用 mDNS）

## 架构

```
VICAP (摄像头)
  ├── CHN0 ──bind──> VO (LCD/HDMI 预览)
  └── CHN1 ──bind──> VENC (H.264) ──> WebRTC ──> 浏览器
                                          ▲
                                   HTTP 信令 (offer/answer)
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `main.c` | 入口、信令处理、H.264 帧转发 |
| `http_server.c/.h` | HTTP 服务器（单线程、CORS） |
| `mpp_pipeline.c/.h` | MPP 管线（VB/VICAP/VO/VENC） |
| `web_page.h` | 嵌入式前端页面 |
| `Makefile` | 构建脚本 |

## 依赖

- [libpeer](../../libs/3rd-party/libpeer) — WebRTC（含 mbedTLS、cJSON、SRTP、SCTP）
- K230 MPP SDK — VB、VICAP、VO、VENC、connector

## License

遵循 K230 RT-Smart SDK 许可协议。
