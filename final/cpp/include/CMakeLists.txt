cmake_minimum_required(VERSION 3.19)
project("wheels") # Project name

set(target_name "wheels") # Target name (target can be an executable or a library)
set(sources
    geometry.cpp
    # Sources
)

set(CMAKE_C_FLAGS_DEBUG "-g -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O2 -w -DNDEBUG")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -w -DNDEBUG")

# # Set this in Makefile and use `make debug|release` instead.
# # Default target in Makefile should be set to debug.
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")

# # Build an executable:
# add_executable(${target_name} "${sources}")
# Or, build a library to be linked:
add_library(${target_name} STATIC ${sources})
# add_library(${target_name} SHARED ${sources})
# add_library(${target_name} MODULE ${sources})  # I have no idea what this is

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target_name} ${OpenCV_LIBS})

# # Magick++ (ImageMagick)
# find_package(ImageMagick COMPONENTS Magick++)
# include_directories(${ImageMagick_INCLUDE_DIRS})
# target_link_libraries(${target_name}
# ${ImageMagick_LIBRARIES}
# MagickCore-7.Q16HDRI # Use this jointly with -DMAGICKCORE_QUANTUM_DEPTH=16.
# )
# add_definitions(-DMAGICKCORE_QUANTUM_DEPTH=16)
# add_definitions(-DMAGICKCORE_HDRI_ENABLE=1)

# Custom headers, add this after all other dependencies
# include_directories("include")

# vim: set ft=cmake:

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Feb 27 2021, 16:21 [CST]
