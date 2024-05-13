# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_library(
  NAME
    testing
  SRCS
    "iree/testing/__init__.py"
    "iree/testing/requirements.py"
)

iree_target_cmake_name(NAME ::testing RESULT TESTING_CMAKE_TARGET)
add_dependencies(iree-test-deps ${TESTING_CMAKE_TARGET})
