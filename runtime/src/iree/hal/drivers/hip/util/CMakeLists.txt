# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    hip_util
  HDRS
    "queue.h"
    "tree.h"
  SRCS
    "queue.c"
    "tree.c"
  DEPS
    iree::base
  PUBLIC
)

iree_cc_test(
  NAME
    iree_hal_hip_util_queue_test
  SRCS
    "queue_test.cc"
  DEPS
    ::hip_util
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
  iree_hal_hip_util_tree_test
  SRCS
    "tree_test.cc"
  DEPS
    ::hip_util
    iree::testing::gtest
    iree::testing::gtest_main
)
