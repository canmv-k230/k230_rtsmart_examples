# 指尖OCR文本识别

## 1.简介

指尖OCR文本识别，识别两只手食指之间的区域中的文本内容。

## 2.应用使用说明

### 2.1 使用帮助

```shell
 #单图推理示例：
./sq_handkp_ocr.elf hand_det.kmodel handkp_ocr_img.jpg 0.15 0.4 handkp_det.kmodel ocr_det_int16.kmodel 0.15 0.4 ocr_rec_int16.kmodel 0

 #视频流推理：（handkpflower_isp.sh）
./sq_handkp_ocr.elf hand_det.kmodel None 0.15 0.4 handkp_det.kmodel ocr_det_int16.kmodel 0.15 0.4 ocr_rec_int16.kmodel 0
```
