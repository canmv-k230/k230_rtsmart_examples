# 编译并运行 Tracker 应用

进入当前目录复制到 `~/canmv_k230/src/rtsmart/examples/multi_object_tracking` 下,选择要使用的算法，并进入对应的算法目录，执行编译命令，这里以botsort为例：

```bash
# 你可以从我们提供的算法中选择，包括：deepsort / bytetrack / ocsort / botsort
cd botsort
./build_app.sh
```

编译完成后，生成的文件将位于：
`~/canmv_k230/src/rtsmart/examples/multi_object_tracking/botsort_track_app/k230_bin/`。

给开发板上电，并将 `k230_bin` 目录拷贝到开发板中。你可以查看磁盘，找到已挂载的 `CanMV` 盘符，然后将 `k230_bin` 复制到 `CanMV/sdcard` 目录下。通过串口连接进行调试，在串口控制台中输入：

```bash
cd /sdcard/k230_bin
./run.sh
```

程序启动后，你应该可以在屏幕上看到视频输出。

如果你想使用 HDMI 显示器进行显示，请修改源文件 `~/canmv_k230/src/rtsmart/examples/multi_object_tracking/botsort_track_app/src/setting.h`，将：

```c
#define DISPLAY_TYPE 'st7701'
```

修改为：

```c
#define DISPLAY_TYPE 'lt9611'
```

然后重新执行上述应用的编译流程。
