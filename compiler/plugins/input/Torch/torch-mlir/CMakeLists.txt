# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
    "${TORCH_MLIR_ROOT_DIR}/include")

iree_cc_library(
  NAME
    defs
  INCLUDES
    "${TORCH_MLIR_ROOT_DIR}/include"
)

###############################################################################
# Torch dialect
###############################################################################

file(GLOB _TorchDialectIR_SRCS "${TORCH_MLIR_ROOT_DIR}/lib/Dialect/Torch/IR/*.cpp")
iree_cc_library(
  NAME
    TorchDialectIR
  SRCS
    ${_TorchDialectIR_SRCS}
  DEPS
    ::defs
    ::TorchDialectGen
    ::TorchDialectTypesGen
    ::TorchDialectUtils
    MLIRBytecodeOpInterface
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
)

file(GLOB _TorchDialectPasses_SRCS "${TORCH_MLIR_ROOT_DIR}/lib/Dialect/Torch/Transforms/*.cpp")
iree_cc_library(
  NAME
    TorchDialectPasses
  SRCS
    ${_TorchDialectPasses_SRCS}
  DEPS
    ::defs
    ::TorchDialectIR
    ::TorchDialectTransformsGen
    ::TorchDialectUtils
    MLIRIR
    MLIRPass
    MLIRTransforms
)

file(GLOB _TorchDialectUtils_SRCS "${TORCH_MLIR_ROOT_DIR}/lib/Dialect/Torch/Utils/*.cpp")
iree_cc_library(
  NAME
    TorchDialectUtils
  SRCS
    ${_TorchDialectUtils_SRCS}
  DEPS
    ::defs
    ::TorchDialectGen
    ::TorchDialectTypesGen
    MLIRIR
    MLIRSupport
)

iree_tablegen_library(
  NAME
    TorchDialectGen
  TD_FILE
    "${TORCH_MLIR_ROOT_DIR}/include/torch-mlir/Dialect/Torch/IR/TorchOps.td"
  OUTS
    -gen-dialect-decls Dialect/Torch/IR/TorchDialect.h.inc
    -gen-dialect-defs Dialect/Torch/IR/TorchDialect.cpp.inc
    -gen-op-decls Dialect/Torch/IR/TorchOps.h.inc
    -gen-op-defs Dialect/Torch/IR/TorchOps.cpp.inc
)

iree_tablegen_library(
  NAME
    TorchDialectTypesGen
  TD_FILE
    "${TORCH_MLIR_ROOT_DIR}/include/torch-mlir/Dialect/Torch/IR/TorchTypes.td"
  OUTS
    -gen-typedef-decls Dialect/Torch/IR/TorchTypes.h.inc
    -gen-typedef-defs Dialect/Torch/IR/TorchTypes.cpp.inc
)

iree_tablegen_library(
  NAME
    TorchDialectTransformsGen
  TD_FILE
    "${TORCH_MLIR_ROOT_DIR}/include/torch-mlir/Dialect/Torch/Transforms/Passes.td"
  OUTS
    -gen-pass-decls Dialect/Torch/Transforms/Passes.h.inc
    -gen-pass-capi-header Dialect/Torch/Transforms/Transforms.capi.h.inc
    -gen-pass-capi-impl Dialect/Torch/Transforms/Transforms.capi.cpp.inc
)

###############################################################################
# TorchConversion dialect
###############################################################################

file(GLOB _TorchConversionDialectIR_SRCS "${TORCH_MLIR_ROOT_DIR}/lib/Dialect/TorchConversion/IR/*.cpp")
iree_cc_library(
  NAME
    TorchConversionDialectIR
  SRCS
    ${_TorchConversionDialectIR_SRCS}
  DEPS
    ::defs
    ::TorchConversionDialectGen
    ::TorchDialectIR
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
)

iree_tablegen_library(
  NAME
    TorchConversionDialectGen
  TD_FILE
    "${TORCH_MLIR_ROOT_DIR}/include/torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.td"
  OUTS
    -gen-dialect-decls -dialect=torch_c Dialect/TorchConversion/IR/TorchConversionDialect.h.inc
    -gen-dialect-defs -dialect=torch_c Dialect/TorchConversion/IR/TorchConversionDialect.cpp.inc
    -gen-op-decls Dialect/TorchConversion/IR/TorchConversionOps.h.inc
    -gen-op-defs Dialect/TorchConversion/IR/TorchConversionOps.cpp.inc
)

iree_tablegen_library(
  NAME
    TorchConversionDialectTransformsGen
  TD_FILE
    "${TORCH_MLIR_ROOT_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms/Passes.td"
  OUTS
    -gen-pass-decls Dialect/TorchConversion/Transforms/Passes.h.inc
    -gen-pass-capi-header Dialect/TorchConversion/Transforms/Transforms.capi.h.inc
    -gen-pass-capi-impl Dialect/TorchConversion/Transforms/Transforms.capi.cpp.inc
)

###############################################################################
# Torch conversion pipelines
# Note that we do not include TorchToStableHLO here due to dependency issues.
# Also, The "TorchDialect" transforms seem to have a circular dependency on
# these conversions. To simplify, we just have this one conversion library
# with all conversion sources and those from Dialect/TorchConversion/Transforms.
###############################################################################

file(GLOB _ConversionPasses_SRCS
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchToArith/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchToLinalg/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchConversionToMLProgram/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchToSCF/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchToTensor/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchToTosa/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/TorchToTMTensor/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/Utils/*.cpp"
  "${TORCH_MLIR_ROOT_DIR}/lib/Dialect/TorchConversion/Transforms/*.cpp"
)
iree_cc_library(
  NAME
    ConversionPasses
  SRCS
    "${TORCH_MLIR_ROOT_DIR}/lib/Conversion/Passes.cpp"
    ${_ConversionPasses_SRCS}
  DEPS
    ::defs
    ::TorchConversionDialectIR
    ::TorchConversionDialectTransformsGen
    ::TorchDialectIR
    ::ConversionPassesGen
    torch-mlir-dialects::TMTensorDialectIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMLProgramDialect
    MLIRSCFDialect
    MLIRTosaDialect
)

iree_tablegen_library(
  NAME
    ConversionPassesGen
  TD_FILE
    "${TORCH_MLIR_ROOT_DIR}/include/torch-mlir/Conversion/Passes.td"
  OUTS
    -gen-pass-decls Conversion/Passes.h.inc
    -gen-pass-capi-header Conversion/Passes.capi.h.inc
    -gen-pass-capi-impl Conversion/Passes.capi.cpp.inc
)

###############################################################################
# CAPI
###############################################################################

file(GLOB _CAPI_SRCS
  "${TORCH_MLIR_ROOT_DIR}/lib/CAPI/*.cpp"
)
# TODO: The way that torch-mlir is doing registration is overly broad.
# It may not be necessary: IREE's registration is already accounting for
# plugins, which includes torch-mlir.
list(FILTER _CAPI_SRCS EXCLUDE REGEX Registration.cpp)
iree_cc_library(
  NAME
    CAPI
  SRCS
    ${_CAPI_SRCS}
  DEPS
    ::defs
    ::ConversionPasses
    ::TorchConversionDialectIR
    ::TorchDialectIR
    ::TorchDialectPasses
    MLIRCAPIIR
    MLIRIR
    MLIRSupport
)
