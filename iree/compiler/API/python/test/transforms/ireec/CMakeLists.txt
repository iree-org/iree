# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_test(
  NAME
    compiler_options_test
  SRCS
    "compiler_options_test.py"
)

iree_py_test(
  NAME
    compile_sample_module
  SRCS
    "compile_sample_module.py"
)
