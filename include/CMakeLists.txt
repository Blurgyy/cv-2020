cmake_minimum_required(VERSION 3.19)
project("") # Project name

set(CMAKE_C_FLAGS_DEBUG "-g -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
# set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# # Set this in Makefile and use `make debug|release` instead.
# # Default target in Makefile should be set to debug.
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")

# # Build an executable:
# add_executable(${target_name} "${sources}")
# # Or, build a library to be linked:
add_library(pa2 STATIC pa2.cpp)
# add_library(${target_name} SHARED)
# add_library(${target_name} MODULE)  # I have no idea what this is.

# vim: set ft=cmake:

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Dec 24 2020, 09:30 [CST]
