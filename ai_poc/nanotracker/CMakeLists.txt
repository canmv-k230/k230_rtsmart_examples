# set common files
set(common_files "../common_files")
include_directories(${common_files})
set(common_files_src 
    ${common_files}/ai_base.cc
    ${common_files}/ai_utils.cc
    ${common_files}/video_pipeline.cc
)

# set source_code
set(src main.cc nanotrack_crop.cc nanotrack_src.cc nanotrack_tracker.cc)
set(all_src ${src} ${common_files_src})
set(bin nanotracker.elf)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(.)

add_executable(${bin} ${all_src})
target_link_libraries(${bin} -Wl,--start-group rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230 sys vicap vb cam_device cam_engine
 hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi 3a buffer_management cameric_drv video_in virtual_hal start_engine cmd_buffer
 switch cameric_reg_drv t_database_c t_mxml_c t_json_c t_common_c vo connector sensor atomic dma -Wl,--end-group)

target_link_libraries(${bin}  -Wl,--start-group opencv_imgcodecs opencv_imgproc opencv_core zlib libjpeg-turbo libopenjp2 libpng libtiff libwebp csi_cv -Wl,--end-group)

install(TARGETS ${bin} DESTINATION bin)