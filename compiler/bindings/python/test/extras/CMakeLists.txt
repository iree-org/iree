# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_INPUT_TORCH)
  # These only import a module and define no tests for pytest to collect.
  iree_py_test(
    NAME
      fx_importer_test
    SRCS
      "fx_importer_test.py"
    SCRIPT
  )

  iree_py_test(
    NAME
      onnx_importer_test
    SRCS
      "onnx_importer_test.py"
    SCRIPT
  )
endif()
