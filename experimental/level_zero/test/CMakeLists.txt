# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_test(
  NAME
    level_zero_test
  SRCS
    level_zero_test.cc
  DEPS
    iree::runtime
    iree::testing::gtest
    iree::testing::gtest_main
  PUBLIC
)
