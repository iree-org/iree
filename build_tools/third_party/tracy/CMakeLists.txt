# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
cmake_minimum_required(VERSION 3.16)
message(STATUS "Start building tracy capture tool")

option(NO_ISA_EXTENSIONS "Disable ISA extensions (don't pass -march=native or -mcpu=native to the compiler)" OFF)
option(NO_STATISTICS "Disable calculation of statistics" ON)
option(NO_PARALLEL_STL "Disable parallel STL" OFF)

set(TRACY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/tracy")
include(${TRACY_SOURCE_DIR}/cmake/version.cmake)


set(CMAKE_CXX_STANDARD 20)

project(
    IREETracyCaptureServer
    LANGUAGES C CXX
    VERSION ${TRACY_VERSION_STRING}
)

include(${TRACY_SOURCE_DIR}/cmake/config.cmake)
include(${TRACY_SOURCE_DIR}/cmake/vendor.cmake)
include(${TRACY_SOURCE_DIR}/cmake/server.cmake)

set(PROGRAM_FILES
    ${TRACY_SOURCE_DIR}/capture/src/capture.cpp
)

add_executable(${PROJECT_NAME} ${PROGRAM_FILES} ${COMMON_FILES} ${SERVER_FILES})

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    OUTPUT_NAME "iree-tracy-capture"
)

target_link_libraries(${PROJECT_NAME} PRIVATE TracyServer TracyGetOpt)
set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
