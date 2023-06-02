cmake_minimum_required(VERSION 3.0)
project(tensorRT8.6_toturial)
set(CMAKE_BUILD_TYPE "Debug")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(tensorRT8.6_toturial ${PROJECT_SOURCE_DIR})

# add_subdirectory(src out)