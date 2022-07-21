# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(NOT IREE_BUILD_COMPILER OR NOT IREE_BUILD_TESTS)
  return()
endif()

iree_bytecode_module(
  NAME
    add_dynamic
  SRC
    "add_dynamic.mlir"
  C_IDENTIFIER
    "iree_tflite_testdata_add_dynamic"
  FLAGS
    "--iree-native-bindings-support=false"
    "--iree-tflite-bindings-support"
    "--iree-hal-target-backends=vmvx"
  PUBLIC
  TESTONLY
)

iree_bytecode_module(
  NAME
    add_multi
  SRC
    "add_multi.mlir"
  C_IDENTIFIER
    "iree_tflite_testdata_add_multi"
  FLAGS
    "--iree-native-bindings-support=false"
    "--iree-tflite-bindings-support"
    "--iree-hal-target-backends=vmvx"
  PUBLIC
  TESTONLY
)

iree_bytecode_module(
  NAME
    add_static
  SRC
    "add_static.mlir"
  C_IDENTIFIER
    "iree_tflite_testdata_add_static"
  FLAGS
    "--iree-native-bindings-support=false"
    "--iree-tflite-bindings-support"
    "--iree-hal-target-backends=vmvx"
  PUBLIC
  TESTONLY
)
