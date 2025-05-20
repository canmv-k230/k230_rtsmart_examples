# 轻量化人脸识别示例

## 概述

人脸识别是计算机视觉领域的一个重要人物，在许多领域都有广泛的应用，如安防、智能家居、医疗等。本示例使用570KB的retinaface检测模型和2.65MB的mobilefacenet特征提取模型，以及C++实现的人脸识别算法的部署开发，在K230实现了一个轻量级的人脸识别系统。包括人脸注册、人脸识别、注册人数查询和注册数据清空功能，实现了完整的人脸识别步骤。

## 编译

```bash
cd rtsmart/examples/face_recognition
./build_app.sh
```

生成的编译产物在 `k230_bin` 目录中。

## 运行

将编译产物拷贝到k230开发板上，运行命令如下：

```bash
# ./face_recognition.elf face_detection_320.kmodel 0.6 0.2 face_recognition.kmodel 75 face_database 0
./run.sh
```

运行参数：

```bash
Usage: face_recognition.elf <kmodel_det> <det_thres> <nms_thres> <kmodel_recg> <recg_thres> <db_dir> <debug_mode>
```

| 参数        | 说明                             | 取值范围     |
| ----------- | -------------------------------- | ------------ |
| kmodel_det  | 人脸检测kmodel路径               | kmodel 路径         |
| det_thres   | 人脸检测阈值                     | 0.0~1.0      |
| nms_thres   | 人脸检测nms阈值                  | 0.0~1.0      |
| kmodel_recg | 人脸识别kmodel路径               | kmodel 路径         |
| recg_thres  | 人脸识别阈值                     | 0~100    |
| db_dir      | 数据库目录                       | 数据库目录路径         |
| debug_mode  | 是否需要调试，0、1、2分别表示不调试、简单调试、详细调试 | 0、1、2 |

## 功能支持

| 功能           | 支持情况 |命令|
| -------------- | -------- |---|
| 打印帮助说明       | ✔        |h/help|
| dump注册帧       | ✔        |i|
| 清空人脸数据库   | ✔        |d|
| 人脸注册         | ✔        |输入人脸名称|
| 注册人数查询     | ✔        |n|
| 退出程序         | ✔        |q|

> 注：
> 注册截图时请确保画面中仅有一张清晰可见的人脸。
> 姓名应使用可识别英文字符，避免特殊符号。
