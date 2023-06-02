PROJECT(1_HelloWorld)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE "Debug")
# tensortRT
set(TENSORRT_DIR "/HardDisk/DEV/SDK/TensorRT-8.6.1.6")
set(CMAKE_PREFIX_PATH ${TENSORRT_DIR})
find_library(NVINFER NAMES libnvinfer.so)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
if(NVINFER)
   message("TensorRT is available!")
   message("NVINFER: ${NVINFER}")
   message("NVPARSERS: ${NVPARSERS}")
   message("NVONNXPARSERS: ${NVONNXPARSERS}")
   set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()
include_directories(${TENSORRT_DIR}/include)

#opencv
set(OpenCV_DIR "/HardDisk/DEV/SDK/opencv-4.7.0/build")
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA headers: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Found CUDA libraries: ${CUDA_LIBRARIES}")
include_directories( ${CUDA_INCLUDE_DIRS} )

add_executable(1_HelloWorld ${PROJECT_SOURCE_DIR}/src/main.cpp)

if(OpenCV_FOUND)
  target_link_libraries(1_HelloWorld ${OpenCV_LIBS})
endif()

if(CUDA_FOUND)
target_link_libraries(1_HelloWorld ${CUDA_LIBS})
endif()

target_link_libraries(1_HelloWorld ${NVINFER})