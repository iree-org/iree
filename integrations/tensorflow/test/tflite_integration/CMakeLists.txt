# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


iree_py_library(
  NAME
    test_util
  SRCS
    "test_util.py"
)

iree_py_test(
  NAME
    mobilenet_v1
  SRCS
    "mobilenet_v1_test.py"
)

iree_py_test(
  NAME
    posenet_i8
  SRCS
    "posenet_i8_test.py"
)
