# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# if(${IREE_BUILD_STATIC_LIBRARY_SAMPLES} AND NOT ${IREE_BUILD_SAMPLES})
#   return()
# endif()

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}
  OR NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER}
  OR NOT ${IREE_BUILD_SAMPLES})
  return()
endif()

iree_bytecode_module(
  NAME
    simple_mul
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_samples_static_library_simple_mul"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
    "-iree-llvm-static-library-output-path=${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o"
  PUBLIC
  TESTONLY
)

iree_cc_binary(
  NAME
    static_library_demo
  SRCS
    "static_library_demo.c"
  DEPS
    ::simple_mul_c
    iree::runtime
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::static_library_loader
    iree::hal::local::task_driver
    iree::modules::hal
    iree::task::api
    iree::vm
    iree::vm::bytecode_module
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
