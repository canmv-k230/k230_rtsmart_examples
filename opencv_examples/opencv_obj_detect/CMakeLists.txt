set(src opencv_obj_detect.cpp)
set(bin opencv_obj_detect.elf)

add_executable(${bin} ${src})
target_link_libraries(${bin} sys)
target_link_libraries(${bin} opencv_objdetect opencv_imgcodecs opencv_imgproc opencv_core libjpeg-turbo libopenjp2 libpng libtiff libwebp zlib csi_cv)

install(TARGETS ${bin} DESTINATION bin)