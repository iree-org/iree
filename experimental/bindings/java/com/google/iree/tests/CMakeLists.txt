# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_BUILD_COMPILER})
  iree_bytecode_module(
    NAME
      simple_mul_bytecode_module
    SRC
      "simple_mul.mlir"
    C_IDENTIFIER
      "iree_java_testdata_simple_mul_bytecode_module"
    FLAGS
      "-iree-mlir-to-vm-bytecode-module"
      "-iree-hal-target-backends=vmla"
  )

  iree_cc_binary(
    NAME
      integration_test
     SRCS
      "integration_test.cc"
    DEPS
       experimental::bindings::java::com::google::iree::native::cc_wrappers
       experimental::bindings::java::com::google::iree::tests::simple_mul_bytecode_module_c
       iree::base::status
  )
endif()
