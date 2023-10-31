# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    common
  HDRS
    "api_impl.h"
    "dylib_entry_point.cc.inc"
    "iree_helpers.h"
    "layout_utils.h"
    "platform.h"
    "tensor_utils.h"
  SRCS
    "api_impl.cc"
    "layout_utils.cc"
    "platform.cc"
    "tensor_utils.cc"
  DEPS
    ::compiler
    iree::base
    iree::hal
    iree::modules::hal
    iree::vm
    iree::vm::bytecode::module
    iree_pjrt_deps::headers
  PUBLIC
)

iree_cc_test(
  NAME
    tensor_utils_test
  SRCS
    "tensor_utils_test.cc"
  DEPS
    ::common
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
    NAME
      compiler
    HDRS
      "compiler.h"
    SRCS
      "hlo_partitioner.cc"
      "iree_compiler.cc"
    DEPS
      ::debugging
      iree::compiler::bindings::c::headers
      iree::compiler::bindings::c::loader
      iree_pjrt::partitioner_api
      iree_pjrt::partitioner_api::loader
    PUBLIC
)

iree_cc_library(
    NAME
      debugging
    HDRS
      "debugging.h"
    SRCS
      "debugging.cc"
    PUBLIC
)

iree_cc_library(
    NAME
      dylib_platform
    HDRS
      "dylib_platform.h"
    SRCS
      "dylib_platform.cc"
    DEPS
      ::common
      ::debugging
      iree::base::internal::file_io
      iree::base::internal::path
    PUBLIC
)
