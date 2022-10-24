# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_build_tools_py_test(
  NAME
    cmake_rule_generator_test
  SRC
    "cmake_rule_generator_test.py"
)

add_subdirectory(device_specs)
