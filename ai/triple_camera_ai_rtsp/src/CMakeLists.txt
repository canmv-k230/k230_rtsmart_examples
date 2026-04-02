# set source_code
set(src main.cc video_pipeline.cc ai_base.cc ai_utils.cc encoded_rtsp_server.cc yolov8_detect.cc)
set(bin triple_cam_ai_rtsp.elf)

set(k230_mpp_root "$ENV{MPP_SRC_DIR}")
set(k230_sdk_build "$ENV{CUR_MPP_BUILD_DIR}")


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)
include_directories(${k230_mpp_root}/userapps/api/)
include_directories(${k230_mpp_root}/include)
include_directories(${k230_mpp_root}/include/comm)
include_directories(${k230_mpp_root}/middleware/src/rtsp_server/include)
link_directories(${nncase_sdk_root}/riscv64/rvvlib/)
link_directories(${k230_sdk_build}/rtsmart/mpp/middleware/lib)

add_executable(${bin} ${src})
target_link_libraries(${bin} -Wl,--start-group rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230 sys vicap vb isp vo venc connector sensor atomic gsdma rtsp_server liveMedia groupsock BasicUsageEnvironment UsageEnvironment -Wl,--end-group)

target_link_libraries(${bin} -Wl,--start-group opencv_imgcodecs opencv_imgproc opencv_core zlib libjpeg-turbo libopenjp2 libpng libtiff libwebp csi_cv -Wl,--end-group)
install(TARGETS ${bin} DESTINATION bin)