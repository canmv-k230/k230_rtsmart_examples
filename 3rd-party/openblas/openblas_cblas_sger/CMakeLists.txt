set(src openblas_cblas_sger.cpp)
set(bin openblas_cblas_sger.elf)

add_executable(${bin} ${src})
target_link_libraries(${bin} sys)
target_link_libraries(${bin} openblas)

install(TARGETS ${bin} DESTINATION bin)