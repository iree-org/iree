# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_TARGET_BACKEND_DYLIB_LLVM_AOT} OR NOT ${IREE_HAL_DRIVER_DYLIB})
  return()
endif()

# Build the bytecode from the mnist.mlir in iree/samples/models.
iree_bytecode_module(
  NAME
    mnist_bytecode_module
  SRC
    "../models/mnist.mlir"
  C_IDENTIFIER
    "mnist_bytecode_module_c"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
  PUBLIC
)

iree_cc_binary(
  NAME
    iree-run-mnist-module
  OUT
    iree-run-mnist-module
  SRCS
    "iree-run-mnist-module.c"
  DATA
    "iree/samples/vision/mnist_test.png"
  DEPS
    ::mnist_bytecode_module_c
    iree::runtime
    iree::tools::utils::image_util
)

iree_lit_test(
  NAME
    iree_run_mnist_module_test
  TEST_FILE
    "mnist_test.txt"
  TOOLS
    ::iree-run-mnist-module
    FileCheck
  DATA
    "mnist_test.png"
  LABELS
    "hostonly"
)
