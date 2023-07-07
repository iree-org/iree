# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(scotttodd): clean up bazel_to_cmake handling here
#   * this is a cc_binary in Bazel, but `linkshared` fits iree_cc_library better
#   * the output file name is platform-specific, get it with $<TARGET_FILE:>
iree_cc_library(
  NAME
    dynamic_library_test_library.so
  SRCS
    "dynamic_library_test_library.cc"
  TESTONLY
  SHARED
)

iree_c_embed_data(
  NAME
    dynamic_library_test_library
  SRCS
    "$<TARGET_FILE:iree::base::testing::dynamic_library_test_library.so>"
  C_FILE_OUTPUT
    "dynamic_library_test_library_embed.c"
  H_FILE_OUTPUT
    "dynamic_library_test_library_embed.h"
  TESTONLY
  FLATTEN
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_library_test
  SRCS
    "dynamic_library_test.cc"
  DEPS
    ::dynamic_library_test_library
    iree::base
    iree::base::internal::dynamic_library
    iree::base::internal::file_io
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "requires-filesystem"
)
