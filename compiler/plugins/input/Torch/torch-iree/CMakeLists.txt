# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(InputConversion)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRMLProgramDialect
    MLIRPass
    iree::compiler::PluginAPI
    torch-iree::InputConversion
    torch-mlir::TorchConversionDialectIR
    torch-mlir::TorchDialectIR
    torch-mlir::TorchDialectPasses
    torch-mlir::ConversionPasses
    torch-mlir-dialects::TMTensorDialectIR
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    input_torch
  TARGET
    ::registration
)
