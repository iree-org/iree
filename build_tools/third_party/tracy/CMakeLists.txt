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
    libzstd
  )

  if(NOT TRACY_DEPS_FOUND)
    message(STATUS "Could not find Tracy dependencies (Tracy server will not be built).")
    message(STATUS "To build Tracy, install libzstd:")
    message(STATUS "  Ubuntu/Debian: `apt install libzstd-dev`")
    message(STATUS "  MacOS: `brew install zstd`")
    return()
  endif()
endif()

#-------------------------------------------------------------------------------
# Source dependencies
# See https://github.com/wolfpld/tracy/blob/master/cmake/vendor.cmake
#-------------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
  capstone
  GIT_REPOSITORY https://github.com/capstone-engine/capstone.git
  GIT_TAG        97db712c91e964718f9cc300e81b9cf76b31a22e # 6.0.0-Alpha1
  EXCLUDE_FROM_ALL
)
set(CAPSTONE_X86_ATT_DISABLE ON CACHE BOOL "" FORCE)
set(CAPSTONE_ALPHA_SUPPORT OFF CACHE BOOL "" FORCE)
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
set(CAPSTONE_M680X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_EVM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_WASM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BPF_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_RISCV_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SH_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_XTENSA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_MACOS_THIN ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(capstone)

FetchContent_Declare(
  PPQSort
  GIT_REPOSITORY https://github.com/GabTux/PPQSort.git
  GIT_TAG        4b964020d67b435dae7ebac7b8f5ecea1f421c58 # v1.0.3
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
      ${TRACY_SOURCE_DIR}/imgui
      ${TRACY_DEPS_INCLUDE_DIRS}
      ${capstone_SOURCE_DIR}/include/capstone
  )
  target_link_libraries(${name}
    PRIVATE
      ${TRACY_DEPS_LIBRARIES}
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
      capstone
      PPQSort
  )
  target_link_directories(${name}
    PRIVATE
      ${TRACY_DEPS_LIBRARY_DIRS}
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
