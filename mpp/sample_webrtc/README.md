# sample_webrtc

K230 RT-Smart 上的 WebRTC 局域网摄像头 Demo。板载摄像头采集 → 硬件编码（H.264/H.265）→ WebRTC 推流到浏览器，零拷贝。

## 功能

- H.264 / H.265 视频实时推流（CBR，编码类型/分辨率/码率可配）
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

# 3. 运行（默认 H.264）
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
| `-t` | 视频编码类型：`h264` / `h265` | `h264` |
| `-W` | 编码宽度 | `1280` |
| `-H` | 编码高度 | `720` |
| `-b` | 编码码率 (kbps) | `2000` |

### 使用示例

```bash
# H.264 默认参数
sample_webrtc.elf -s 2 -c 605274512

# H.265 编码，1080p
sample_webrtc.elf -t h265 -s 2 -c 605274512 -W 1920 -H 1080 -b 4000

# H.264 + 自定义码率
sample_webrtc.elf -t h264 -s 2 -c 605274512 -b 3000
```

## 浏览器要求

### mDNS 隐私保护

Chrome/Edge 默认启用 mDNS 隐私保护，会隐藏本地 IP 导致 WebRTC 连接失败。页面会自动检测并提示。

修复方式（任选其一）：

1. 地址栏输入 `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns` → 设为 **Disabled** → 重启
2. 启动参数加 `--disable-features=WebRtcHideLocalIpsWithMdns`
3. 使用 Firefox（默认未启用 mDNS）

### H.265 浏览器兼容性

> **注意**：使用 `-t h265` 时，部分浏览器无法解码 H.265 (HEVC) WebRTC 流。

| 浏览器 | H.265 WebRTC 支持 | 说明 |
|--------|-------------------|------|
| Chrome | ✅ 支持 | 需硬件 HEVC 解码支持（大部分桌面端已具备） |
| Edge | ✅ 支持 | 与 Chrome 同内核，同样依赖硬件解码 |
| Firefox | ❌ 不支持 | WebRTC 尚未实现 HEVC 解码 |
| Safari | ⚠️ 部分支持 | macOS/iOS 上依赖硬件解码，Windows 版不支持 |

**建议**：
- 如需最大兼容性，使用默认的 H.264 编码（`-t h264`）
- 仅在确认客户端浏览器支持 H.265 时使用 `-t h265`
- H.265 在相同画质下码率更低，适合带宽受限但对客户端可控的场景

## 架构

```
VICAP (摄像头)
  ├── CHN0 ──bind──> VO (LCD/HDMI 预览)
  └── CHN1 ──bind──> VENC (H.264/H.265) ──> WebRTC ──> 浏览器
                                                  ▲
                                           HTTP 信令 (offer/answer)
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `main.c` | 入口、信令处理、编码帧转发 |
| `http_server.c/.h` | HTTP 服务器（单线程、CORS） |
| `mpp_pipeline.c/.h` | MPP 管线（VB/VICAP/VO/VENC） |
| `web_page.h` | 嵌入式前端页面 |
| `Makefile` | 构建脚本 |

## 依赖

- [libpeer](../../libs/3rd-party/libpeer) — WebRTC（含 mbedTLS、cJSON、SRTP、SCTP）
- K230 MPP SDK — VB、VICAP、VO、VENC、connector

## License

遵循 K230 RT-Smart SDK 许可协议。
