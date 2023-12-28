# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_ALL_CTS_TESTS
  "allocator"
  "buffer_mapping"
  "command_buffer"
  "command_buffer_dispatch"
  "command_buffer_push_constants"
  "descriptor_set_layout"
  "driver"
  "event"
  "executable_cache"
  "file"
  "pipeline_layout"
  "semaphore"
  "semaphore_submission"
  PARENT_SCOPE
)

# These tests use executables produced by the iree-compile compiler tool.
# If the compiler is disabled or a HAL driver implementation is not yet
# connected to a functional compiler target, these tests can be skipped.
set(IREE_EXECUTABLE_CTS_TESTS
  "command_buffer_dispatch"
  "command_buffer_push_constants"
  "executable_cache"
  PARENT_SCOPE
)

# List of testdata/{name}.mlir source files.
set(IREE_ALL_CTS_EXECUTABLE_SOURCES
  "command_buffer_dispatch_test"
  "command_buffer_push_constants_test"
  "executable_cache_test"
  PARENT_SCOPE
)

iree_cc_library(
  NAME
    cts_test_base
  HDRS
    "cts_test_base.h"
  DEPS
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
  PUBLIC
)

iree_cc_library(
  NAME
    allocator_test_library
  HDRS
    "allocator_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    buffer_mapping_test_library
  HDRS
    "buffer_mapping_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    command_buffer_test_library
  HDRS
    "command_buffer_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    command_buffer_dispatch_test_library
  HDRS
    "command_buffer_dispatch_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    command_buffer_push_constants_test_library
  HDRS
    "command_buffer_push_constants_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    descriptor_set_layout_test_library
  HDRS
    "descriptor_set_layout_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    driver_test_library
  HDRS
    "driver_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    event_test_library
  HDRS
    "event_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    executable_cache_test_library
  HDRS
    "executable_cache_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    file_test_library
  HDRS
    "file_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    pipeline_layout_test_library
  HDRS
    "pipeline_layout_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    semaphore_test_library
  HDRS
    "semaphore_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)

iree_cc_library(
  NAME
    semaphore_submission_test_library
  HDRS
    "semaphore_submission_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
  TESTONLY
)
