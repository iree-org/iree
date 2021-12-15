# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_ALL_CTS_TESTS
  "allocator"
  "buffer_mapping"
  "command_buffer"
  "descriptor_set_layout"
  "descriptor_set"
  "driver"
  "event"
  "executable_layout"
  "semaphore_submission"
  "semaphore"
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
)

iree_cc_library(
  NAME
    descriptor_set_test_library
  HDRS
    "descriptor_set_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
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
)

iree_cc_library(
  NAME
    executable_layout_test_library
  HDRS
    "executable_layout_test.h"
  DEPS
    ::cts_test_base
    iree::base
    iree::hal
    iree::testing::gtest
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
)
