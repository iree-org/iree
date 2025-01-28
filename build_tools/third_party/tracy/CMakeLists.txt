# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.16.3)

project(IREETracyServer C CXX)

set(TRACY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/tracy")

find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
# Detect package manager
#-------------------------------------------------------------------------------

message(STATUS "Checking for Tracy dependencies...")
find_program(PKGCONFIG pkg-config)
if(NOT PKGCONFIG)
  message(STATUS "Could not find pkg-config to get dependencies; you need to specify them manually or use pkg-config")
  message(STATUS "  Ubuntu/Debian: `apt install pkg-config`")
  message(STATUS "  MacOS: `brew install pkg-config`")
else()
  include(FindPkgConfig)

  # Deps slightly differ by platform but some are common.
  pkg_check_modules(TRACY_DEPS
    tbb
    libzstd
  )
  pkg_check_modules(TRACY_CAPSTONE_DEPS
    capstone
  )

  if(NOT TRACY_DEPS_FOUND)
    message(STATUS "Could not find Tracy dependencies (Tracy server will not be built).")
    message(STATUS "To build Tracy, install packages libzstd, and tbb:")
    message(STATUS "  Ubuntu/Debian: `apt install libcapstone-dev libtbb-dev libzstd-dev`")
    message(STATUS "  MacOS: `brew install capstone tbb zstd`")
    return()
  endif()

  if(NOT TRACY_CAPSTONE_DEPS_FOUND)
    message(STATUS "Could not find capstone, a Tracy dependency (Tracy server will not be built).")
    message(STATUS "To build Tracy, install capstone or build from source:")
    message(STATUS "  Ubuntu/Debian: `apt install libcapstone-dev`")
    message(STATUS "  MacOS: `brew install capstone`")
    return()
  endif()
endif()

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

function(setup_cxx_options name)
  set_target_properties(${name}
    PROPERTIES
      CXX_STANDARD 17
  )
  target_compile_options(${name}
    PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-unused-result>
  )
  target_include_directories(${name}
    PUBLIC
      ${TRACY_SOURCE_DIR}/imgui
      ${TRACY_DEPS_INCLUDE_DIRS}
      ${TRACY_CAPSTONE_DEPS_INCLUDE_DIRS}
      # capstone-next moved capstone.h to a capstone/ subdirectory, but the
      # pkg-config isn't updated yet as of April 2022.
      ${TRACY_CAPSTONE_DEPS_INCLUDE_DIRS}/capstone
  )
  target_link_libraries(${name}
    PRIVATE
      ${TRACY_DEPS_LIBRARIES}
      ${TRACY_CAPSTONE_DEPS_LIBRARIES}
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
  )
  target_link_directories(${name}
    PRIVATE
      ${TRACY_DEPS_LIBRARY_DIRS}
      ${TRACY_CAPSTONE_DEPS_LIBRARY_DIRS}
  )
endfunction()

#-------------------------------------------------------------------------------
# Common library
#-------------------------------------------------------------------------------

file(GLOB COMMON_SRCS ${TRACY_SOURCE_DIR}/public/common/*.cpp)
add_library(IREETracyCommon
  ${COMMON_SRCS}
)
setup_cxx_options(IREETracyCommon)

#-------------------------------------------------------------------------------
# Server library
#-------------------------------------------------------------------------------

file(GLOB SERVER_SRCS ${TRACY_SOURCE_DIR}/server/*.cpp)
add_library(IREETracyServer
  ${SERVER_SRCS}
)
setup_cxx_options(IREETracyServer)
target_link_libraries(IREETracyServer
  PRIVATE
    IREETracyCommon
)

#-------------------------------------------------------------------------------
# Standalone capture server
#-------------------------------------------------------------------------------

add_executable(IREETracyCaptureServer
  ${TRACY_SOURCE_DIR}/capture/src/capture.cpp
)
set_target_properties(IREETracyCaptureServer
  PROPERTIES
    OUTPUT_NAME "iree-tracy-capture"
)
setup_cxx_options(IREETracyCaptureServer)
target_link_libraries(IREETracyCaptureServer
  PRIVATE
    IREETracyCommon
    IREETracyServer
)
