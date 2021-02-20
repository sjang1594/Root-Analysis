cmake_minimum_required(VERSION 3.16)
project(autocropper)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
ADD_COMPILE_OPTIONS(-g -Wall)

# Find all executables
file(GLOB project_SRCS src/*.cpp src/*.h)

# Add project executable
add_executable(autocropper ${project_SRCS})

target_link_libraries(autocropper ${OpenCV_LIBS})