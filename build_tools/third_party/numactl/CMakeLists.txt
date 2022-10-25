# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
cmake_minimum_required(VERSION 3.16.3)

project(IREENuma C CXX)

set(NUMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/numactl")

message(STATUS "Found numactl!")

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
      ${NUMA_SOURCE_DIR}
  )
  target_link_libraries(${name}
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
  )
  target_link_directories(${name}
    PRIVATE
      ${NUMA_SOURCE_DIR}
  )
endfunction()

file(GLOB COMMON_SRCS ${NUMA_SOURCE_DIR}/*.c)

add_library(IREENumaCommon
  ${COMMON_SRCS}
)

setup_cxx_options(IREENumaCommon)