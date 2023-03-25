# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: I don't yet know how to make this compatible with bazel. When it is,
# use bazel_to_cmake to generate.

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "print.mlir"
  TOOLS
    FileCheck
    iree-opt
)
