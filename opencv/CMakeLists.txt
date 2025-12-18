cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
	message(FATAL_ERROR "OpenCV not found. Install libopencv-dev or set OpenCV_DIR to your OpenCV installation.")
endif()

add_executable(my_program  deleteblack.cpp)
target_include_directories(my_program PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(my_program PRIVATE ${OpenCV_LIBS})