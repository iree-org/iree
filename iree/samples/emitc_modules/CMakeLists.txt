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
      iree::base::cc
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::cc
      iree::vm::ops
      iree::vm::shims_emitc
  )

  iree_cc_test(
    NAME
      import_module_test
    SRCS
      "import_module_test.cc"
    DEPS
      ::add_module
      ::import_module_a
      ::import_module_b
      iree::base
      iree::base::cc
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::cc
      iree::vm::ops
      iree::vm::shims_emitc
  )

  iree_c_module(
    NAME
      import_module_a
    SRC
      "import_module_a.mlir"
    H_FILE_OUTPUT
      "import_module_a.h"
  )

  iree_c_module(
    NAME
    import_module_b
    SRC
      "import_module_b.mlir"
    H_FILE_OUTPUT
      "import_module_b.h"
  )
endif()
