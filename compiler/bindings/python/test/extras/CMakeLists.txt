# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_INPUT_TORCH)
  iree_py_test(
    NAME
      fx_importer_test
    SRCS
      "fx_importer_test.py"
  )

  iree_py_test(
    NAME
      onnx_importer_test
    SRCS
      "onnx_importer_test.py"
  )
endif()
