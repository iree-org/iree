# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The mnist builder depends on onnx, which needs the torch input support.
if(IREE_INPUT_TORCH)
  iree_py_test(
    NAME
      mnist_builder_test
    SRCS
      "mnist_builder_test.py"
  )
endif()
