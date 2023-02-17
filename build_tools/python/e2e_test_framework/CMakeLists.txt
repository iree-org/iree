# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_build_tools_py_test(
  NAME
    serialization_test
  SRC
    "serialization_test.py"
)

iree_build_tools_py_test(
  NAME
    unique_ids_test
  SRC
    "unique_ids_test.py"
)

add_subdirectory(device_specs)
