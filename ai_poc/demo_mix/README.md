# 应用串烧

## 1.简介

本项目实现了多个demo串烧操作，是手势关键点识别、动态手势识别、人脸姿态角和单目标跟踪的集成。可以作为智能跟踪拍摄车的软件部分实现隔空调整底盘位置，隔空调整相机角度，追踪人脸目标。

**注意：本示例仅支持HDMI显示方式**  

## 2.应用使用说明

### 2.1使用帮助

#### 2.1.1 手势说明

支持手势如下：

![支持手势](https://kendryte-download.canaan-creative.com/k230/downloads/doc_images/ai_demo/demo_mix/gesture.jpg)

one手势进入动态手势识别，love手势退出；yeah手势进入姿态角调整，会选择距离屏幕中心点最近的点调整，love手势退出；three手势进入单目标追踪，会选择距离屏幕中心点最近的人脸进行跟踪。

#### 2.1.2串口通信

**请您按照开发板自行配置串口，并修改代码中的串口设备。**  

协议格式如图所示：

![协议格式](https://kendryte-download.canaan-creative.com/k230/downloads/doc_images/ai_demo/demo_mix/data_frame.png)

接收到的十六进制数据两位为一个字节，前两位表示帧头AA，最后两位表示帧尾BB，第3，4位表示设备编号，0为底盘，1为相机；第5、6位表示命令编号，设备0有5类命令（0，1，2，3，4），设备1有2类命令（0，1）；第7、8位表示数据长度，单位是字节；剩余数据为数据。

注意：数据解析时使用int8格式范围为-128~127；

#### 2.1.3 动态手势识别命令

| 指令（16进制） | 说明                                                         | 解释                 |
| -------------- | ------------------------------------------------------------ | -------------------- |
| AA00000100BB   | frame head:AA <br/>device_num:00  <br/>command_num:00   <br/>data_length:01  <br/>data:00 <br/>frame tail:BB | 底盘小车开始向前移动 |
| AA00010101BB   | frame head:AA <br/>device_num:00  <br/>command_num:01  <br/>data_length:01  <br/>data:01 <br/>frame tail:BB | 底盘小车开始向左移动 |
| AA00020102BB   | frame head:AA <br/>device_num:00  <br/>command_num:02  <br/>data_length:01  <br/>data:02 <br/>frame tail:BB | 底盘小车开始向后移动 |
| AA00030103BB   | frame head:AA <br/>device_num:00  <br/>command_num:03  <br/>data_length:01  <br/>data:03 <br/>frame tail:BB | 底盘小车开始向右移动 |
| AA00040104BB   | frame head:AA <br/>device_num:00  <br/>command_num:04  <br/>data_length:01  <br/>data:04 <br/>frame tail:BB | 底盘小车停止移动     |

接收到指令后会一直移动，直到收到middle命令才能继续下一个方向的调整，因此不同方向的调整命令必须由middle作为停止命令分开。

#### 2.1.4 人脸姿态角命令

| 指令（16进制）   | 说明                                                         | 解释                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AA010003xxxxxxBB | frame head:AA <br/>device_num:01  <br/>command_num:00   <br/>data_length:03  <br/>data:xxxxxx <br/>frame tail:BB | data数据每两位解析成一个int8数据（-128~127），分别表示与x、z、y轴的正方向的夹角 |

姿态角的y,z,x的示意图如下：

![姿态角](https://kendryte-download.canaan-creative.com/k230/downloads/doc_images/ai_demo/demo_mix/pose.png)

#### 2.1.5 单目标人脸跟踪命令

| 指令（16进制） | 说明                                                         | 解释                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AA0101020000BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0000 <br/>frame tail:BB | data数据占4位，前两位表示左右偏向，后两位表示上下偏向；0000表示跟踪正常，停止小车移动和相机转动 |
| AA0101020100BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0100 <br/>frame tail:BB | 0100，相机向左转动，直到出现0000停住                         |
| AA0101020200BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0200 <br/>frame tail:BB | 0200，相机向右转动，直到出现0000停住                         |
| AA0101020001BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0001 <br/>frame tail:BB | 0001，相机向下转动，直到出现0000停住                         |
| AA0101020002BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0002 <br/>frame tail:BB | 0002，相机向上转动，直到出现0000停住                         |
| AA0101020101BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0101 <br/>frame tail:BB | 0101，相机向左下转动，直到出现0000停住                       |
| AA0101020102BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0102 <br/>frame tail:BB | 0102，相机向左上转动，直到出现0000停住                       |
| AA0101020201BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0201 <br/>frame tail:BB | 0201，相机向右下转动，直到出现0000停住                       |
| AA0101020202BB | frame head:AA <br/>device_num:01  <br/>command_num:01   <br/>data_length:02  <br/>data:0202 <br/>frame tail:BB | 0202，相机向右上转动，直到出现0000停住                       |

#### 2.1.6 快速启动

连接好串口，在PC端运行如下python脚本，可以在命令行查看接收到的数据，注意替换串口ID：

```python
import serial
import time


def decode_command(hex_values):
    device_num = hex_values[0]
    command_num = hex_values[1]
    data_len = hex_values[2]
    data = hex_values[3:]
    if device_num == 0:
        if command_num == 0:
            print("正在将底盘向前移动")
        elif command_num == 1:
            print("正在将底盘向左移动")
        elif command_num == 2:
            print("正在将底盘向后移动")
        elif command_num == 3:
            print("正在将底盘向右移动")
        elif command_num == 4:
            print("停止移动底盘")
        else:
            print("非法命令")
    elif device_num == 1:
        if command_num == 0:
            print("姿态角夹角数据为：y轴(pitch):" + str(data[0]) + " z轴(roll):" + str(data[1]) + " x轴(yaw):" + str(data[2]))
        elif command_num == 1:
            if data[0] == 0 and data[1] == 0:
                print("跟踪正常")
            elif data[0] == 1 and data[1] == 0:
                print("镜头向左运动")
            elif data[0] == 2 and data[1] == 0:
                print("镜头向右运动")
            elif data[0] == 0 and data[1] == 1:
                print("镜头向下运动")
            elif data[0] == 0 and data[1] == 2:
                print("镜头向上运动")
            elif data[0] == 1 and data[1] == 1:
                print("镜头向左、向下运动")
            elif data[0] == 1 and data[1] == 2:
                print("镜头向左、向上运动")
            elif data[0] == 2 and data[1] == 1:
                print("镜头向右、向下运动")
            elif data[0] == 2 and data[1] == 2:
                print("镜头向右、向上运动")
            else:
                print("非法命令")
        else:
            print("非法命令")
    else:
        print("非法命令")


def main():
    # 串口设置
    serial_port = "COM55"
    baud_rate = 115200

    # 打开串口
    ser = serial.Serial(serial_port, baud_rate)
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    try:
        while True:
            # 从串口读取数据
            data = ser.read(ser.in_waiting)

            if data == b'':
                pass
            else:
                # 处理接收到的数据
                print("串口接收十六进制数据:", data.hex())
                hex_string = str(data.hex())
                # 将十六进制字符串分割为两个字符一组
                hex_values = [hex_string[i:i + 2] for i in range(0, len(hex_string), 2)]

                # 将每个十六进制值转换为有符号 int8 类型
                int8_values = [(int(hex_value, 16) + 128) % 256 - 128 for hex_value in hex_values]
                # 将命令解析成动作
                decode_command(int8_values)
            time.sleep(0.05)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 信号，用于停止程序
        print("Program terminated by user.")

    finally:
        # 关闭串口
        ser.close()


if __name__ == "__main__":
    main()

```

无串口连接的情况下可以正常运行，不查看串口数据时可以不连串口。

在大核执行AI线程：

```shell
# 准备kmodel列表：hand_det.kmodel、handkp_det.kmodel、gesture.kmodel、face_detection_320.kmodel、face_pose.kmodel、cropped_test127.kmodel、nanotrack_backbone_sim.kmodel、nanotracker_head_calib_k230.kmodel
#准备文件：shang.bin、xia.bin、zuo.bin、you.bin、demo_mix.elf
#视频流推理（demo_mix.sh）
./demo_mix.elf
```
