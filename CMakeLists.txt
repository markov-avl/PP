cmake_minimum_required(VERSION 3.29)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mavx -mavx2 -march=native -mtune=native")

add_executable(Lecture01 lecture-01.cpp)
add_executable(Lecture02 lecture-02.cpp)
add_executable(Lecture03 lecture-03.cpp)
