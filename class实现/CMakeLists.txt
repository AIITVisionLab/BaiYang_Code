cmake_minimum_required(VERSION 3.10)
project(StudentProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main 
    main.cpp
    Student.cpp
    MyString.cpp
)
