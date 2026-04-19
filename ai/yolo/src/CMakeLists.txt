set(src main.cc pipeline.cc ai_base.cc yolov5.cc yolov8.cc yolo11.cc yolo26.cc utils.cc)
set(bin yolo.elf)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)
include_directories(${k230_mpp_root}/userapps/api/)
include_directories(${k230_mpp_root}/include)
include_directories(${k230_mpp_root}/include/comm)
include_directories(${k230_mpp_root}/userapps/sample/sample_vo)
link_directories(${nncase_sdk_root}/riscv64/rvvlib/)

add_executable(${bin} ${src})
target_link_libraries(${bin} -Wl,--start-group rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230 sys vicap vb isp vo connector sensor atomic gsdma -Wl,--end-group)

target_link_libraries(${bin} -Wl,--start-group opencv_imgcodecs opencv_imgproc opencv_core zlib libjpeg-turbo libopenjp2 libpng libtiff libwebp csi_cv -Wl,--end-group)
install(TARGETS ${bin} DESTINATION bin)