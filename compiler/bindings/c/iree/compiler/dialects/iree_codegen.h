// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECTS_IREE_CODEGEN_H
#define IREE_COMPILER_DIALECTS_IREE_CODEGEN_H

#include "mlir-c/AffineMap.h"
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

MLIR_CAPI_EXPORTED void
ireeCodegenGetExecutableVariantOps(MlirModule module, size_t *numOps,
                                   MlirOperation *executableOps);

MLIR_CAPI_EXPORTED void ireeCodegenGetTunerRootOps(MlirModule module,
                                                   size_t *numOps,
                                                   MlirOperation *rootOps);

struct ireeCodegenAttentionOpDetail {
  MlirAttribute batch;
  MlirAttribute m;
  MlirAttribute k1;
  MlirAttribute k2;
  MlirAttribute n;
  int64_t domainRank;
};

MLIR_CAPI_EXPORTED ireeCodegenAttentionOpDetail
ireeCodegenGetAttentionOpDetail(MlirAffineMap qMap, MlirAffineMap kMap,
                                MlirAffineMap vMap, MlirAffineMap oMap);

MLIR_CAPI_EXPORTED bool
ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op);

struct ireeCodegenIGEMMGenericConvDetails {
  // Indexing maps for IGEMM contraction (ArrayAttr of AffineMapAttr).
  MlirAttribute igemmContractionMaps;
  // Loop bounds for IGEMM indexing (I64ArrayAttr).
  MlirAttribute igemmLoopBounds;
  // Loop iterator types for IGEMM indexing (ArrayAttr of StringAttr).
  MlirAttribute igemmLoopIterators;
  // Output permutation for im2col tensor (I64ArrayAttr).
  MlirAttribute im2colOutputPerm;
  // Filter reassociation indices (ArrayAttr of ArrayAttr of I64).
  MlirAttribute filterReassocIndices;
  // True if output channel precedes spatial dims, causing filter and input
  // operands to be swapped in igemmContractionMaps/igemmOperands.
  bool isOutputChannelFirst;
  // Mapping from convolution dimensions to IGEMM dimensions (ArrayAttr of
  // [conv_dim, igemm_dim] pairs). Many-to-one: multiple conv dims can map to
  // the same IGEMM dim. IGEMM dims are indices into igemmLoopBounds/Iterators.
  MlirAttribute convToIgemmDimMap;
};

// Checks if IGEMM generic convolution details can be queried for the given
// operation.
MLIR_CAPI_EXPORTED bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op);

// Gets IGEMM generic convolution details for the given operation.
MLIR_CAPI_EXPORTED ireeCodegenIGEMMGenericConvDetails
ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op);

struct ireeCodegenScaledContractionDimensions {
  MlirAttribute batch;
  MlirAttribute m;
  MlirAttribute n;
  MlirAttribute k;
  MlirAttribute kB;
};

MLIR_CAPI_EXPORTED bool
ireeCodegenMlirOperationIsAScaledContractionOp(MlirOperation op);

MLIR_CAPI_EXPORTED ireeCodegenScaledContractionDimensions
ireeCodegenInferScaledContractionDimensions(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_DIALECTS_IREE_CODEGEN_H
