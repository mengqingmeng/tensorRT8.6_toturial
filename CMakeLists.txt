PROJECT(tensorRT8.6_toturial)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE "Debug")

#include_directories(${PROJECT_SOURCE_DIR}/include)

#add_executable(tensorRT8.6_toturial ${PROJECT_SOURCE_DIR}/src/main.cpp)

# if(OpenCV_FOUND)
#   target_link_libraries(tensorRT8.6_toturial ${OpenCV_LIBS})
# endif()

# if(CUDA_FOUND)
# target_link_libraries(tensorRT8.6_toturial ${CUDA_LIBS})
# endif()

# target_link_libraries(tensorRT8.6_toturial ${NVINFER})

add_subdirectory(1_HelloWorld)