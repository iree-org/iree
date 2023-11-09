# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Set the root for package namespacing to the current directory.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/experimental/hip")

iree_add_all_subdirs()

if(NOT DEFINED ROCM_HEADERS_API_ROOT)
  set(ROCM_HEADERS_API_ROOT "${IREE_SOURCE_DIR}/third_party/hip-build-deps/include")
endif()

if(NOT EXISTS "${ROCM_HEADERS_API_ROOT}/hip/hip_version.h")
  message(SEND_ERROR "Could not find HIP headers at: ${ROCM_HEADERS_API_ROOT}")
endif()

iree_cc_library(
  NAME
    hip
  HDRS
    "api.h"
  SRCS
    "api.h"
    "hip_driver.c"
  INCLUDES
    "${ROCM_HEADERS_API_ROOT}"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::hal
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
    "status_util.h"
  TEXTUAL_HDRS
    "dynamic_symbol_tables.h"
  SRCS
    "hip_headers.h"
    "dynamic_symbols.c"
    "status_util.c"
  INCLUDES
    "${ROCM_HEADERS_API_ROOT}"
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal::dynamic_library
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    "dynamic_symbols_test.cc"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=hip"
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
)
