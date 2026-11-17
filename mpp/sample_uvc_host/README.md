# UVC host sample

```
sample_uvc_host.elf <connector_type> <rotation> <fourcc> <width> <height> <total_frame> [device=0]
```

The optional final argument selects `/dev/video0`, `/dev/video1`, etc. Existing
commands without it select device zero. For example, with standard 1080p HDMI
output, capture 300 MJPEG frames from the second USB camera:

```sh
/sdcard/sample_uvc_host.elf 759104028 0 MJPEG 640 480 300 1
```

Set the SDK configuration `CONFIG_USBHOST_MAX_VIDEO_CLASS=2` to keep two cameras
attached. The 01Studio CanMV defconfig enables this. One camera can be opened at
a time; stop the first capture before starting the other. Device indexes follow
USB enumeration order and can change when cameras are unplugged and reattached.
The `uvc_host_formats` shell command lists the registered device paths and modes.

CanMV selects a camera with `UVC.select_device(index)` before `probe()`,
`list_video_mode()` or `select_video_mode()`. Selection releases its previous
frame, decoder and stream. For C callers, call `uvc_host_exit()` before
`uvc_host_select_device(index)`; selection while initialized returns `-EBUSY`.
The selection persists across `uvc_host_exit()`, `UVC.stop()` and mode
changes, so a restart reopens the same camera. CanMV returns to device zero on
soft reset.

Run this standalone sample after stopping any other media application, including
CanMV's media allocator, so they do not compete over global VB initialization.
