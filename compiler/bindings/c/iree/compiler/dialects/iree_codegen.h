// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECTS_IREE_CODEGEN_H
#define IREE_COMPILER_DIALECTS_IREE_CODEGEN_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// The following C API is **NOT STABLE** and likely to change in the future.
// It mirrors the IREE Codegen Dialect which is not stable itself.

MLIR_CAPI_EXPORTED bool
ireeAttributeIsACodegenDispatchLoweringPassPipelineAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID
ireeCodegenDispatchLoweringPassPipelineAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute ireeCodegenDispatchLoweringPassPipelineAttrGet(
    MlirContext mlirCtx, uint32_t value);

MLIR_CAPI_EXPORTED
uint32_t
ireeCodegenDispatchLoweringPassPipelineAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
ireeAttributeIsACodegenTranslationInfoAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeCodegenTranslationInfoAttrGetTypeID(void);

struct ireeCodegenTranslationInfoParameters {
  MlirAttribute passPipeline;      // DispatchLoweringPassPipelineAttr.
  MlirAttribute codegenSpec;       // Optional SymbolRefAttr.
  const int64_t *workgroupSize;    // Optional ArrayRef<int64_t>.
  size_t numWorkgroupSizeElements; // Size of the ArrayRef above.
  int64_t subgroupSize;            // Optional int64_t.
  MlirAttribute configuration;     // Optional DictionaryAttr.
};

MLIR_CAPI_EXPORTED MlirAttribute ireeCodegenTranslationInfoAttrGet(
    MlirContext mlirCtx, ireeCodegenTranslationInfoParameters parameters);

MLIR_CAPI_EXPORTED ireeCodegenTranslationInfoParameters
ireeCodegenTranslationInfoAttrGetParameters(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
ireeAttributeIsACodegenCompilationInfoAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeCodegenCompilationInfoAttrGetTypeID(void);

struct ireeCodegenCompilationInfoParameters {
  MlirAttribute loweringConfig;
  MlirAttribute translationInfo;
};

MLIR_CAPI_EXPORTED MlirAttribute ireeCodegenCompilationInfoAttrGet(
    MlirContext mlirCtx, ireeCodegenCompilationInfoParameters parameters);

MLIR_CAPI_EXPORTED ireeCodegenCompilationInfoParameters
ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_DIALECTS_IREE_CODEGEN_H
