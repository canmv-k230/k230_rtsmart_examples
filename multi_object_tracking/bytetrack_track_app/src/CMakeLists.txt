set(bytetrack_dir "${CMAKE_CURRENT_SOURCE_DIR}/../bytetrack")
include_directories(${bytetrack_dir})

file(GLOB bytetrack_files_src
    "${bytetrack_dir}/*.cpp"
)


set(src main.cc ai_base.cc ai_utils.cc video_pipeline.cc yolov8_det.cc ${bytetrack_files_src})
set(bin bytetrack_track.elf)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)
include_directories(${k230_mpp_root}/userapps/api/)
include_directories(${k230_mpp_root}/include)
include_directories(${k230_mpp_root}/include/comm)
include_directories(${k230_mpp_root}/userapps/sample/sample_vo)
link_directories(${nncase_sdk_root}/riscv64/rvvlib/)
include_directories(../botsort)
include_directories(../../libs/Eigen32)

add_executable(${bin} ${src})
target_link_libraries(${bin} -Wl,--start-group rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230 sys vicap vb cam_device cam_engine
 hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi 3a buffer_management cameric_drv video_in virtual_hal start_engine cmd_buffer
 switch cameric_reg_drv t_database_c t_mxml_c t_json_c t_common_c vo connector sensor atomic dma -Wl,--end-group)

target_link_libraries(${bin} opencv_imgcodecs opencv_imgproc opencv_core zlib libjpeg-turbo libopenjp2 libpng libtiff libwebp csi_cv)

install(TARGETS ${bin} DESTINATION bin)

add_compile_options(-ffunction-sections -fdata-sections)

add_link_options(-Wl,--gc-sections)