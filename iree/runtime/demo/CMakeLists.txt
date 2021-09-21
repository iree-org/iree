# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is manually updated to link against aggregate libraries instead
# of in-tree individual libraries as a test that they are built properly and
# complete. See rules under iree/libraries/CMakeLists.txt

if (NOT ${IREE_BUILD_COMPILER} OR NOT ${IREE_BUILD_TESTS})
  return()
endif()

iree_cc_binary(
  NAME
    hello_world_embedded
  SRCS
    "hello_world_explained.c"
  DEFINES
    "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_EMBEDDED_DATA"
  DEPS
    iree::libraries::runtime_base
    iree::libraries::runtime_core_drivers
    iree::runtime::testdata::simple_mul_module_c
)

iree_run_binary_test(
  NAME
    "hello_world_embedded_test"
  TEST_BINARY
    ::hello_world_embedded
)

iree_cc_binary(
  NAME
    hello_world_file
  SRCS
    "hello_world_explained.c"
  DEFINES
    "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_COMMAND_LINE_ARG"
  DEPS
    iree::libraries::runtime_base
    iree::libraries::runtime_core_drivers
)

iree_cc_binary(
  NAME
    hello_world_terse
  SRCS
    "hello_world_terse.c"
  DEPS
    iree::libraries::runtime_base
    iree::libraries::runtime_core_drivers
    iree::runtime::testdata::simple_mul_module_c
)

iree_run_binary_test(
  NAME
    "hello_world_terse_test"
  TEST_BINARY
    ::hello_world_terse
)
