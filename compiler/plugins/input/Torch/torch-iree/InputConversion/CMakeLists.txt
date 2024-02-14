# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(test)

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
    PassHeaders
  HDRS
    "PassDetail.h"
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
    MLIRTransforms
  PUBLIC
)

iree_cc_library(
  NAME
    InputConversion
  HDRS
    "Passes.h"
  SRCS
    "BitCastQuantTensor.cpp"
    "ConvertTMTensorToLinalgExt.cpp"
    "SetStrictSymbolicShapes.cpp"
    "Passes.cpp"
  DEPS
    ::PassHeaders
    ::PassesIncGen
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRTensorDialect
    MLIRTransforms
    torch-mlir::ConversionPasses
    torch-mlir::TorchConversionDialectIR
    torch-mlir::TorchDialectPasses
    torch-mlir-dialects::TMTensorDialectIR
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::LinalgExt::IR
  PUBLIC
)
