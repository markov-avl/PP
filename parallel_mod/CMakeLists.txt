cmake_minimum_required(VERSION 3.29)
project(parallel_mod)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64")

include_directories(src)

add_executable(
        parallel_mod
        src/config.h
        src/mod_ops.cpp
        src/mod_ops.h
        src/num_threads.cpp
        src/num_threads.h
        src/performance.cpp
        src/performance.h
        src/randomize.cpp
        src/randomize.h
        src/test.cpp
        src/test.h
        src/vector_mod.h
        src/vector_mod.cpp
        src/entrypoint.cpp
)
