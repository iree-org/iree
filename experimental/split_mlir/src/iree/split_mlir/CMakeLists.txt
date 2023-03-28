# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    split_mlir_lib
  HDRS
    "Passes.h"
    "Passes.h.inc"
  SRCS
    "Passes.cpp"
  DEPS
    ::PassesIncGen
    MLIRFuncDialect
    MLIRIR
    MLIRPass
  PUBLIC
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    ::split_mlir_lib
    MLIRIR
    MLIRPass
    iree::compiler::PluginAPI
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    split_mlir
  TARGET
    ::registration
)
