# sample_webrtc

K230 RT-Smart 上的 WebRTC 局域网摄像头 Demo。板载摄像头采集 → 硬件编码（H.264/H.265）→ WebRTC 推流到浏览器，零拷贝。

## 功能

- H.264 / H.265 视频实时推流（CBR，编码类型/分辨率/码率可配）
- LCD / HDMI 本地预览
- 浏览器一键连接，自带 Web UI
- SDP offer/answer 信令（HTTP）
- 自动处理浏览器 mDNS 隐私候选地址
- 自动按浏览器访问的本机接口发布 ICE 地址，支持 Wi-Fi SoftAP 模式
- 最多 4 个浏览器并发观看，每个客户端使用独立的 ICE/DTLS/SRTP 会话
- 使用设备局域网 IP 直接打开 Web 页面和信令接口

## 快速开始

```bash
# 1. menuconfig 启用 libpeer 和 sample_webrtc
#    RT-Smart UserSpace Libraries Configuration  →  Enable Build libpeer (WebRTC)  = Y
#    RT-Smart UserSpace Examples Configuration   →  Enable MPP examples           = Y
#                                                   →  Enable Build sample_webrtc   = Y
make menuconfig

# 2. 编译（在 K230 RT-Smart SDK 根目录）
make

# 3. 运行（默认 H.265、512 kbps、无音频）
sample_webrtc.elf -p 8080 -s 2 -c 605274512 -W 1280 -H 720

# 4. 浏览器访问程序启动日志中打印的 URL
# http://<设备IP>:8080/
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p` | HTTP 端口 | `8080` |
| `-s` | CSI 设备号 (0-2) | `2` |
| `-c` | 显示接口：`605274512`=LCD, `757006876`=HDMI | LCD |
| `-t` | 视频编码类型：`h264` / `h265` | `h265` |
| `-W` | 编码宽度 | `1280` |
| `-H` | 编码高度 | `720` |
| `-b` | 编码码率 (kbps) | `512` |

### 使用示例

```bash
# H.265 默认参数
sample_webrtc.elf -s 2 -c 605274512

# H.265 编码，1080p
sample_webrtc.elf -t h265 -s 2 -c 605274512 -W 1920 -H 1080 -b 4000

# H.264 + 自定义码率
sample_webrtc.elf -t h264 -s 2 -c 605274512 -b 3000

```

## 浏览器要求

### mDNS 隐私保护

Chrome/Edge 可以保持默认的 mDNS 隐私保护设置。信令服务器会使用 HTTP 连接的客户端地址解析浏览器的 `.local` ICE 候选地址，无需修改浏览器 flags 或启动参数。

在 Wi-Fi AP 模式下，服务端会从浏览器的 HTTP 连接获取实际使用的本机 AP 地址，并将该地址写入 SDP host candidate，不依赖默认路由。

启动信息会为每个可用的 SoftAP、STA 和 LAN IPv4 接口分别打印完整访问 URL。连接到开发板 AP 的浏览器应使用 SoftAP 子网对应的 URL。

HTTP 服务监听所有本机接口。浏览器连接后，服务端从已接受的 HTTP socket 获取本次连接使用的本机地址，并自动将 ICE UDP socket 和 SDP host candidate 绑定到同一个地址。

每个浏览器在 `GET /offer` 时获得独立的会话 ID 和 `PeerConnection`。`POST /answer` 使用该会话 ID 路由到对应连接，因此连接到 SoftAP 和 STA 接口的浏览器可以同时观看，不会重新绑定或关闭其他客户端的 ICE socket。编码器只编码一次，VENC 线程把同一编码帧发送到所有已连接会话。

默认最多允许 4 个并发客户端。客户端主动断开时 Web UI 会释放会话；信令未完成或 ICE 保活超时的会话也会自动清理。达到上限时，新的 `/offer` 请求返回 HTTP `503`。

Web 页面和信令接口不要求访问令牌，HTTP 响应仍不允许跨域访问。该 Demo 使用明文 HTTP，适合受信任的局域网测试，不应直接暴露到公网。需要跨不可信网络部署时，应在前端增加 HTTPS 和正式的身份认证。

### H.265 浏览器兼容性

> **注意**：使用 `-t h265` 时，部分浏览器无法解码 H.265 (HEVC) WebRTC 流。

| 浏览器 | H.265 WebRTC 支持 | 说明 |
|--------|-------------------|------|
| Chrome | ✅ 支持 | 需硬件 HEVC 解码支持（大部分桌面端已具备） |
| Edge | ✅ 支持 | 与 Chrome 同内核，同样依赖硬件解码 |
| Firefox | ❌ 不支持 | WebRTC 尚未实现 HEVC 解码 |
| Safari | ⚠️ 部分支持 | macOS/iOS 上依赖硬件解码，Windows 版不支持 |

**建议**：
- 默认使用 H.265 以降低码率和带宽占用
- 如需最大浏览器兼容性，使用 H.264 编码（`-t h264`）
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
| `main.c` | 入口、多客户端会话管理、信令处理、编码帧转发 |
| `http_server.c/.h` | HTTP 服务器（单线程、同源信令、安全响应头） |
| `mpp_pipeline.c/.h` | MPP 管线（VB/VICAP/VO/VENC） |
| `web_page.h` | 嵌入式前端页面 |
| `Makefile` | 构建脚本 |

## 依赖

- [libpeer](../../libs/3rd-party/libpeer) — WebRTC（含 mbedTLS、cJSON、SRTP、SCTP）
- K230 MPP SDK — VB、VICAP、VO、VENC、connector

## License

遵循 K230 RT-Smart SDK 许可协议。
