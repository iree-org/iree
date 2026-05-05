# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of special logic for the benchmark library.

iree_add_all_subdirs()

if(IREE_ENABLE_THREADING)
  iree_cc_library(
    NAME
      benchmark
    HDRS
      "benchmark.h"
    SRCS
      "benchmark_full.cc"
    DEPS
      benchmark
      iree::base
    PUBLIC
  )
else()
  iree_cc_library(
    NAME
      benchmark
    HDRS
      "benchmark.h"
    SRCS
      "benchmark_nop.c"
    DEPS
      iree::base
    PUBLIC
  )
endif()

iree_cc_library(
  NAME
    benchmark_main
  SRCS
    "benchmark_main.c"
  DEPS
    ::benchmark
    iree::base::tooling::flags
  TESTONLY
  PUBLIC
)

iree_cc_library(
  NAME
    gtest
  HDRS
    "gtest.h"
    "status_matchers.h"
  DEPS
    gmock
    gtest
    iree::base
  TESTONLY
  PUBLIC
)

iree_cc_library(
  NAME
    gtest_main
  SRCS
    "gtest_main.cc"
  DEPS
    ::gtest
    gmock
    gtest
    iree::base
    iree::base::tooling::flags
  TESTONLY
  PUBLIC
)

iree_cc_library(
  NAME
    temp_file
  HDRS
    "temp_file.h"
  SRCS
    "temp_file.cc"
  DEPS
    ::gtest
    iree::base
  TESTONLY
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

iree_cc_library(
  NAME
    coordinated_test
  HDRS
    "coordinated_test.h"
  SRCS
    "coordinated_test.c"
  DEPS
    iree::base
  PUBLIC
)

iree_cc_library(
  NAME
    coordinated_test_main
  SRCS
    "coordinated_test_main.cc"
  DEPS
    ::coordinated_test
    ::gtest
    gtest
    iree::base
    iree::base::tooling::flags
  TESTONLY
  PUBLIC
)

iree_cc_test(
  NAME
    coordinated_test_test
  SRCS
    "coordinated_test_test.cc"
  DEPS
    ::coordinated_test
    ::coordinated_test_main
    ::gtest
  LABELS
    "noriscv"
)
