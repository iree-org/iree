# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_build_tools_py_test(
  NAME
    model_artifacts_test
  SRC
    "model_artifacts_test.py"
)

iree_build_tools_py_test(
  NAME
    iree_artifacts_test
  SRC
    "iree_artifacts_test.py"
)


add_subdirectory(cmake_generator)
