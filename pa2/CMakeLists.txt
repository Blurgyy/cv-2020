cmake_minimum_required(VERSION 3.19)
project("pa2") # Project name

set(target_name "harris") # Target name (target can be an executable or a library)
set(sources "main.cpp") # Sources

set(CMAKE_C_FLAGS_DEBUG "-g -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# # Set this in Makefile and use `make debug|release` instead.
# # Default target in Makefile should be set to debug.
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")

# Build an executable:
add_executable(${target_name} "${sources}")
# # Or, build a library to be linked:
# add_library(${target_name} STATIC)
# add_library(${target_name} SHARED)
# add_library(${target_name} MODULE)  # I have no idea what this is.

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target_name} ${OpenCV_LIBS})

# OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Custom utilities
add_subdirectory(include)
include_directories(include)
target_link_libraries(${target_name} pa2)

# vim: set ft=cmake:

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Dec 24 2020, 09:24 [CST]
