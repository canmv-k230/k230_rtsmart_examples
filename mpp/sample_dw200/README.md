# DW200 sample

The sample reads a raw YUV420SP (NV12) frame, applies the DW200 map, and
writes the output as `channel0_<width>x<height>_<format>.bin`.  DW200 emits
YUV422SP internally; when `output 0` requests YUV420SP (as in the included
configs), the sample runs the VSE conversion before saving the final file.

Run the identity configuration first. It uses zero distortion coefficients, so
the output should match the input apart from hardware processing:

```text
cd /sdcard/app/examples/mpp
./sample_dw200.elf sample_dw200_identity_640x480.json
```

The lens-correction configuration contains illustrative calibration values:

```text
./sample_dw200.elf sample_dw200_lens_correction_640x480.json
```

The command-line API selector covers each driver interface. Running two frames
also exercises the low-level DWE clear-and-restart operation:

```text
./sample_dw200.elf --api low-level --frames 2 sample_dw200_identity_640x480.json
./sample_dw200.elf --api legacy sample_dw200_identity_640x480.json
./sample_dw200.elf --api vdev sample_dw200_identity_640x480.json
./sample_dw200.elf --api all --frames 2 sample_dw200_identity_640x480.json
```

The virtual-device run writes its generated register list to
`dw200_vdev_registers.txt`. Legacy and virtual-device output files have
`_legacy` and `_vdev` suffixes so they can be compared with the low-level
result.

Use the VSE DMA configuration to exercise VSE without a DWE input. This path is
supported by the low-level and virtual-device APIs:

```text
./sample_dw200.elf --api low-level sample_dw200_vse_dma_640x480.json
./sample_dw200.elf --api vdev sample_dw200_vse_dma_640x480.json
```

Those coefficients must be replaced with calibration data for the camera and
resolution being processed. The installed `input_640x480_yuv420sp.bin` is an
NV12 color-bar frame and is useful for checking the data path, but it is not a
camera-distorted image.

The output is a raw frame.  For the included 640x480 NV12 configs, compare it
with an NV12 viewer or convert it using the same width, height, and pixel
format; opening it as RGB or YUV422 will make a correct result look corrupted.
