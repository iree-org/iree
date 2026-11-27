# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# These run their checks when they are imported, which only works for a
# script.
iree_py_test(
  NAME
    registration_test
  SRCS
    "registration_test.py"
  SCRIPT
)

iree_py_test(
  NAME
    dialects_test
  SRCS
    "dialects_test.py"
  SCRIPT
)
