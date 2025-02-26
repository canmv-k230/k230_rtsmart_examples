set(src openblas_fortran_dgemm.cpp)
set(bin openblas_fortran_dgemm.elf)

add_executable(${bin} ${src})
target_link_libraries(${bin} sys)
target_link_libraries(${bin} openblas)

install(TARGETS ${bin} DESTINATION bin)