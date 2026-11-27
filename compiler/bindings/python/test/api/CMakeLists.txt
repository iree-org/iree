# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# On Windows this file disables itself when it is imported and then defines
# no tests, which only works for a script.
iree_py_test(
  NAME
    api_test
  SRCS
    "api_test.py"
  SCRIPT
)

# The next two run their checks when they are imported, which only works for
# a script.
iree_py_test(
  NAME
    output_buffer_reference_test
  SRCS
    "output_buffer_reference_test.py"
  SCRIPT
)

iree_py_test(
  NAME
    tuner_api_test
  SRCS
    "tuner_api_test.py"
  SCRIPT
)
