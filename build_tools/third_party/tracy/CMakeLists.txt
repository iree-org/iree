# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)

project(IREETracyServer C CXX)

set(TRACY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/tracy")

find_package(Threads REQUIRED)
find_package(Git QUIET)

#-------------------------------------------------------------------------------
# Source dependencies
# See https://github.com/wolfpld/tracy/blob/master/cmake/vendor.cmake
#-------------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
  capstone
  GIT_REPOSITORY https://github.com/capstone-engine/capstone.git
  GIT_TAG        b2bf6327b5c7dc43829130b7ccdb02ef9a65a990 # 6.0.0-Alpha11
  EXCLUDE_FROM_ALL
)
set(CAPSTONE_X86_ATT_DISABLE ON CACHE BOOL "" FORCE)
set(CAPSTONE_ALPHA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_ARC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_HPPA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_LOONGARCH_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_M680X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_M68K_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_MIPS_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_MOS65XX_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_PPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SPARC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SYSTEMZ_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_XCORE_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_TRICORE_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_TMS320C64X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_EVM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_WASM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BPF_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_RISCV_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SH_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_XTENSA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_MACOS_THIN ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(capstone)

FetchContent_Declare(
  zstd
  GIT_REPOSITORY https://github.com/facebook/zstd.git
  GIT_TAG        f8745da6ff1ad1e7bab384bd1f9d742439278e99 # v1.5.7
  SOURCE_SUBDIR  build/cmake
  EXCLUDE_FROM_ALL
)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(zstd)

# Upstream Tracy carries these patches against PPQSort; one fixes a
# binary_semaphore being released more than once. Upstream applies them with
# fuzz (via CPM), so reduce the required context to match.
FetchContent_Declare(
  PPQSort
  GIT_REPOSITORY https://github.com/GabTux/PPQSort.git
  GIT_TAG        21c5a20ca3daf572caaf0a8522db32e27e28fa1d # v1.0.6
  PATCH_COMMAND  ${GIT_EXECUTABLE} apply -C1 --ignore-whitespace
                 ${TRACY_SOURCE_DIR}/cmake/ppqsort-nodebug.patch
                 ${TRACY_SOURCE_DIR}/cmake/ppqsort-semaphore.patch
  UPDATE_DISCONNECTED 1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(PPQSort)

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

function(setup_cxx_options name)
  set_target_properties(${name}
    PROPERTIES
      CXX_STANDARD 20
  )
  target_compile_options(${name}
    PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-unused-result>
  )
  target_include_directories(${name}
    PUBLIC
      ${TRACY_SOURCE_DIR}/public/common
      ${TRACY_SOURCE_DIR}/server
      ${capstone_SOURCE_DIR}/include/capstone
  )
  target_link_libraries(${name}
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
      capstone
      libzstd
      PPQSort
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
# Command line tool utilities
#-------------------------------------------------------------------------------

file(GLOB UTIL_SRCS ${TRACY_SOURCE_DIR}/util/*.cpp)
add_library(IREETracyUtil
  ${UTIL_SRCS}
)
setup_cxx_options(IREETracyUtil)
target_include_directories(IREETracyUtil
  PUBLIC
    ${TRACY_SOURCE_DIR}/util
)
target_link_libraries(IREETracyUtil
  PRIVATE
    IREETracyServer
)

add_library(IREETracyGetOpt
  ${TRACY_SOURCE_DIR}/getopt/getopt.c
)
target_include_directories(IREETracyGetOpt
  PUBLIC
    ${TRACY_SOURCE_DIR}/getopt
)

# The capture tool prints the Tracy git revision it was built from.
set(_TRACY_GIT_REF "unknown")
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} -C ${TRACY_SOURCE_DIR} log -1 --format=%h
    OUTPUT_VARIABLE _TRACY_GIT_REF_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TRACY_GIT_REF_RESULT
    ERROR_QUIET
  )
  if(_TRACY_GIT_REF_RESULT EQUAL 0 AND _TRACY_GIT_REF_OUT)
    set(_TRACY_GIT_REF "${_TRACY_GIT_REF_OUT}")
  endif()
endif()
file(CONFIGURE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/GitRef.hpp"
  CONTENT "#pragma once\nnamespace tracy { static inline const char* GitRef = \"${_TRACY_GIT_REF}\"; }\n"
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
target_include_directories(IREETracyCaptureServer
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(IREETracyCaptureServer
  PRIVATE
    IREETracyCommon
    IREETracyServer
    IREETracyUtil
    IREETracyGetOpt
)
