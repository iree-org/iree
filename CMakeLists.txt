# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(IREE_PJRT)
cmake_policy(SET CMP0069 NEW)

# NOTE: The IREE source code must be cloned to a directory side-by-side with
# this project.
# TODO: Change this when it moves in-tree
set(IREE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../iree)

# Enable LTO if supported.
option(IREE_PJRT_ENABLE_LTO "Enable LTO (link time optimization) if supported" OFF)
include(CheckIPOSupported)
check_ipo_supported(RESULT _ireert_lto_supported OUTPUT error)
if(IREE_PJRT_ENABLE_LTO)
  if(_ireert_lto_supported)
    message(STATUS "Enabling LTO")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO not supported by toolchain bit requested (ignored)")
  endif()
endif()

# Customize defaults.
option(IREE_BUILD_COMPILER "Disable compiler for runtime-library build" OFF)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)

# Include IREE.
message(STATUS "Including IREE from ${IREE_ROOT_DIR}")
add_subdirectory("${IREE_ROOT_DIR}" "iree_core" EXCLUDE_FROM_ALL)

add_subdirectory(src)
