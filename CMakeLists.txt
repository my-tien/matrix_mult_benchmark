project(matrix_mult_benchmark)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
include_directories(/opt/OpenBLAS/include)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_compile_options(${PROJECT_NAME} PRIVATE "-std=c++14" "-pedantic-errors")
target_compile_options(${PROJECT_NAME} PRIVATE "-Wall" "-Wextra")

target_link_libraries (${PROJECT_NAME} /opt/OpenBLAS/lib/libopenblas.a -lpthread)
