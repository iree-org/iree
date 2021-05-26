# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_library(
  NAME
    jax
  SRCS
    "__init__.py"
    "frontend.py"
)

iree_py_test(
  NAME
    frontend_test
  SRCS
    "frontend_test.py"
)
