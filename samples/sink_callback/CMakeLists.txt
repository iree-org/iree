# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_PYTHON_BINDINGS)
  return()
endif()

# These tests use compiler APIs as well as runtime APIs.
if(NOT IREE_BUILD_COMPILER)
  return()
endif()

iree_py_test(
  NAME
    sink_callback_log_simple
  SRCS
    "sink_callback_log.py"
)

iree_py_test(
  NAME
    sink_callback_log_async
  SRCS
    "sink_callback_log_async.py"
)
