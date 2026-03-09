cmake_minimum_required(VERSION 3.16)
project(opencv)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(GENERATED_MAIN "${CMAKE_CURRENT_BINARY_DIR}/generated_main.cpp")
file(WRITE "${GENERATED_MAIN}" "int main(){return 0;}\n")
set_source_files_properties("${GENERATED_MAIN}" PROPERTIES GENERATED TRUE)

# Always start with a valid source so add_executable() never gets an empty list.
set(OPENCV_APP_SOURCES "${GENERATED_MAIN}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
	set(OPENCV_APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/quickdemo.cpp")
	list(APPEND OPENCV_APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/quickdemo.cpp")
endif()

# If any stale/non-existent files got into the list, drop them to avoid configure errors.
foreach(_src IN LISTS OPENCV_APP_SOURCES)
	if(NOT EXISTS "${_src}")
		message(STATUS "Skipping missing source file: ${_src}")
		list(REMOVE_ITEM OPENCV_APP_SOURCES "${_src}")
	endif()
endforeach()

if(NOT OPENCV_APP_SOURCES)
	list(APPEND OPENCV_APP_SOURCES "${GENERATED_MAIN}")
endif()

add_executable(opencv ${OPENCV_APP_SOURCES})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/quickdemo.h")
	target_sources(opencv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/quickdemo.h")
endif()

target_include_directories(opencv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv PRIVATE ${OpenCV_LIBS})
