# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(IREE_BUILD_BINDINGS_TFLITE_JAVA)
  add_subdirectory(java/org/tensorflow/lite)
endif()

iree_cc_library(
  NAME
    shim
  HDRS
    "include/tensorflow/lite/c/c_api.h"
    "include/tensorflow/lite/c/c_api_experimental.h"
    "include/tensorflow/lite/c/common.h"
  SRCS
    "interpreter.c"
    "interpreter.h"
    "model.c"
    "model.h"
    "options.c"
    "options.h"
    "shim.c"
    "shim.h"
    "tensor.c"
    "tensor.h"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::synchronization
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::modules::hal
    iree::vm
    iree::vm::bytecode::module
  PUBLIC
)

if(NOT IREE_BUILD_COMPILER OR NOT IREE_BUILD_TESTS)
  return()
endif()

iree_cc_test(
  NAME
    smoke_test
  SRCS
    "smoke_test.cc"
  DEPS
    ::shim
    iree::runtime::bindings::tflite::testdata::add_dynamic_c
    iree::runtime::bindings::tflite::testdata::add_multi_c
    iree::runtime::bindings::tflite::testdata::add_static_c
    iree::testing::gtest
    iree::testing::gtest_main
)
