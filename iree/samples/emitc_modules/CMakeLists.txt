# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_ENABLE_EMITC})
  iree_c_module(
    NAME
      add_module
    SRC
      "add.mlir"
    H_FILE_OUTPUT
      "add_module.h"
  )

  iree_cc_test(
    NAME
      add_module_test
    SRCS
      "add_module_test.cc"
    DEPS
      ::add_module
      iree::base
      iree::base::status
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::cc
      iree::vm::ops
      iree::vm::shims_emitc
  )
endif()
