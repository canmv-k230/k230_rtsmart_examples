#file(GLOB src *.c)
set (src  spi_main.c)
set(bin spi.elf)

add_executable(${bin} ${src})
# target_link_libraries(${bin} sys)
# target_link_libraries(${bin} opencv_imgcodecs opencv_imgproc opencv_core libjpeg-turbo libopenjp2 libpng libtiff libwebp zlib csi_cv)

install(TARGETS ${bin} DESTINATION spi)
