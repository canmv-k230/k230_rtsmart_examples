set(src opencv_detect_features2d.cpp)
set(bin opencv_detect_features2d.elf)

add_executable(${bin} ${src})
target_link_libraries(${bin} sys)
target_link_libraries(${bin} opencv_features2d opencv_imgcodecs opencv_imgproc opencv_core libjpeg-turbo libopenjp2 libpng libtiff libwebp zlib csi_cv)

install(TARGETS ${bin} DESTINATION bin)