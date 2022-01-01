# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_compiler_api_py_test(
  NAME
    compiler_core_test
  MAIN
    "compiler_core_test.py"
)

iree_compiler_api_py_test(
  NAME
    compiler_tf_test
  MAIN
    "compiler_tf_test.py"
)

iree_compiler_api_py_test(
  NAME
    compiler_tflite_test
  MAIN
    "compiler_tflite_test.py"
)

iree_compiler_api_py_test(
  NAME
    compiler_xla_test
  MAIN
    "compiler_xla_test.py"
)
