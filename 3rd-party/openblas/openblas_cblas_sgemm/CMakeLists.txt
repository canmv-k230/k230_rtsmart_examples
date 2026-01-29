set(src openblas_cblas_sgemm.cpp)
set(bin openblas_cblas_sgemm.elf)

add_executable(${bin} ${src})
target_link_libraries(${bin} sys)
target_link_libraries(${bin} openblas)

install(TARGETS ${bin} DESTINATION bin)