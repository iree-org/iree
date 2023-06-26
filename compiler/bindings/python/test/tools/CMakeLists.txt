# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# These tests perform linking via the Compiler API, which is only supported
# in bundled-LLVM builds at the moment (#14086).
if(IREE_BUILD_BUNDLED_LLVM)
iree_py_test(
  NAME
    compiler_core_test
  SRCS
    "compiler_core_test.py"
)
endif()  # IREE_BUILD_BUNDLED_LLVM

iree_py_test(
  NAME
    compiler_tf_test
  SRCS
    "compiler_tf_test.py"
)

iree_py_test(
  NAME
    compiler_tflite_test
  SRCS
    "compiler_tflite_test.py"
)

