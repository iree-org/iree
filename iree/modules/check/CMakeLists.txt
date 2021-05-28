# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

# Doesn't use bazel_to_cmake because IREE_HAL_DRIVER_VMVX filtering is custom logic
if(${IREE_HAL_DRIVER_VMVX})
  iree_cc_test(
    NAME
      check_test
    SRCS
      "check_test.cc"
    DEPS
      ::native_module
      absl::span
      iree::base
      iree::base::internal
      iree::base::logging
      iree::base::status
      iree::hal
      iree::hal::vmvx::registration
      iree::modules::hal
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::bytecode_module
      iree::vm::cc
  )
endif()


iree_cc_library(
  NAME
    native_module
  HDRS
    "native_module.h"
  SRCS
    "native_module.cc"
  DEPS
    iree::base
    iree::base::internal
    iree::base::status
    iree::hal
    iree::modules::hal
    iree::testing::gtest
    iree::vm
    iree::vm::cc
  TESTONLY
  PUBLIC
)
