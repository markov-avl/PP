cmake_minimum_required(VERSION 3.29)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mavx -mavx2 -march=native -mtune=native")

add_executable(Matrix
        matrix/entrypoint.cpp
        matrix/mul.cpp
        matrix/mul.h
        matrix/sum.cpp
        matrix/sum.h
)

add_executable(ParallelMod
        parallel_mod/config.h
        parallel_mod/mod_ops.cpp
        parallel_mod/mod_ops.h
        parallel_mod/num_threads.cpp
        parallel_mod/num_threads.h
        parallel_mod/performance.cpp
        parallel_mod/performance.h
        parallel_mod/randomize.cpp
        parallel_mod/randomize.h
        parallel_mod/test.cpp
        parallel_mod/test.h
        parallel_mod/vector_mod.h
        parallel_mod/vector_mod.cpp
        parallel_mod/entrypoint.cpp
)
