# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "smoketest.mlir"
  TOOLS
    FileCheck
    iree-opt
)
