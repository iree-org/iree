# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_test(
  NAME
    api_test
  SRCS
    "api_test.py"
)

iree_py_test(
  NAME
    mlir_interop_reference_test
  SRCS
    "mlir_interop_reference_test.py"
)

iree_py_test(
  NAME
    output_buffer_reference_test
  SRCS
    "output_buffer_reference_test.py"
)
