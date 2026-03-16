cmake_minimum_required(VERSION 4.1)
project(smartPoint)

set(CMAKE_CXX_STANDARD 20)

add_executable(main.cpp
        SimpleSharedPtr.cpp
        SimpleSharedPtr.h
        SimpleSharedPtr.cpp
        Student.cpp
        Student.h
)
