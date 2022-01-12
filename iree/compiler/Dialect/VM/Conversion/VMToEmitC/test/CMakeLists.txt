# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "arithmetic_ops.mlir"
    "arithmetic_ops_f32.mlir"
    "arithmetic_ops_i64.mlir"
    "assignment_ops.mlir"
    "assignment_ops_f32.mlir"
    "assignment_ops_i64.mlir"
    "comparison_ops.mlir"
    "comparison_ops_f32.mlir"
    "comparison_ops_i64.mlir"
    "const_ops.mlir"
    "const_ops_f32.mlir"
    "const_ops_i64.mlir"
    "conversion_ops.mlir"
    "conversion_ops_f32.mlir"
    "conversion_ops_i64.mlir"
    "shift_ops.mlir"
    "shift_ops_i64.mlir"
    "type_conversion.mlir"
  DATA
    FileCheck
    iree::tools::iree-opt
)
