# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_build_tools_py_test(
  NAME
    iree_definitions_test
  SRC
    "iree_definitions_test.py"
)

iree_build_tools_py_test(
  NAME
    utils_test
  SRC
    "utils_test.py"
)
