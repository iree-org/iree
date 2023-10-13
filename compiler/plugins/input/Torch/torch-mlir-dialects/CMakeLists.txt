# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_TORCH_MLIR_DIALECTS_DIR "${TORCH_MLIR_ROOT_DIR}/externals/llvm-external-projects/torch-mlir-dialects")
list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
    "${_TORCH_MLIR_DIALECTS_DIR}/include")

iree_cc_library(
  NAME
    defs
  INCLUDES
    "${_TORCH_MLIR_DIALECTS_DIR}/include"
)

###############################################################################
# TMTensor dialect
###############################################################################

file(GLOB _TMTensorDialectIR_SRCS
  "${_TORCH_MLIR_DIALECTS_DIR}/lib/Dialect/TMTensor/IR/*.cpp")
iree_cc_library(
  NAME
    TMTensorDialectIR
  SRCS
    ${_TMTensorDialectIR_SRCS}
  DEPS
    ::defs
    ::TMTensorInterfacesIncGen
    ::TMTensorScalarLoopOpInterfaceIncGen
    ::TMTensorDialectGen
    MLIRAffineDialect
    MLIRDialectUtils
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRViewLikeInterface
)

# TODO: We don't seem to depend on this yet (outside of ref backend), so
# don't include.
# file(GLOB _TMTensorDialectTransforms_SRCS
#   "${_TORCH_MLIR_DIALECTS_DIR}/lib/Dialect/TMTensor/Transforms/*.cpp")
# iree_cc_library(
#   NAME
#     TMTensorDialectTransforms
#   SRCS
#     ${_TMTensorDialectTransforms_SRCS}
#   DEPS
#     ::defs
#     ::TMTensorDialectIR
#     ::TMTensorDialectTransformsGen
#     TorchMLIRTMTensorDialect
#     MLIRAffineDialect
#     MLIRIR
#     MLIRLinalgDialect
#     MLIRLinalgTransforms
#     MLIRMathDialect
#     MLIRMemRefDialect
#     MLIRPass
#     MLIRSCFDialect
#     MLIRFuncDialect
#     MLIRSupport
#     MLIRTensorDialect
#     MLIRTransforms
# )

iree_tablegen_library(
  NAME
    TMTensorInterfacesIncGen
  TD_FILE
    "${_TORCH_MLIR_DIALECTS_DIR}/include/torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorInterfaces.td"
  OUTS
    -gen-op-interface-decls Dialect/TMTensor/IR/TMTensorOpInterfaces.h.inc
    -gen-op-interface-defs Dialect/TMTensor/IR/TMTensorOpInterfaces.cpp.inc
    -gen-type-interface-decls Dialect/TMTensor/IR/TMTensorTypeInterfaces.h.inc
    -gen-type-interface-defs Dialect/TMTensor/IR/TMTensorTypeInterfaces.cpp.inc
)

iree_tablegen_library(
  NAME
    TMTensorScalarLoopOpInterfaceIncGen
  TD_FILE
    "${_TORCH_MLIR_DIALECTS_DIR}/include/torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.td"
  OUTS
    -gen-op-interface-decls Dialect/TMTensor/IR/ScalarLoopOpInterface.h.inc
    -gen-op-interface-defs Dialect/TMTensor/IR/ScalarLoopOpInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    TMTensorDialectGen
  TD_FILE
    "${_TORCH_MLIR_DIALECTS_DIR}/include/torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.td"
  OUTS
    -gen-dialect-decls -dialect=tm_tensor Dialect/TMTensor/IR/TMTensorDialect.h.inc
    -gen-dialect-defs -dialect=tm_tensor Dialect/TMTensor/IR/TMTensorDialect.cpp.inc
    -gen-op-decls Dialect/TMTensor/IR/TMTensorOps.h.inc
    -gen-op-defs Dialect/TMTensor/IR/TMTensorOps.cpp.inc
    -gen-typedef-decls Dialect/TMTensor/IR/TMTensorTypes.h.inc
    -gen-typedef-defs Dialect/TMTensor/IR/TMTensorTypes.cpp.inc
)

iree_tablegen_library(
  NAME
    TMTensorDialectTransformsGen
  TD_FILE
    "${_TORCH_MLIR_DIALECTS_DIR}/include/torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.td"
  OUTS
    -gen-pass-decls Dialect/TMTensor/Transforms/Passes.h.inc
    -gen-pass-capi-header Dialect/TMTensor/Transforms/Passes.capi.h.inc
    -gen-pass-capi-impl Dialect/TMTensor/Transforms/Passes.capi.cpp.inc
)
