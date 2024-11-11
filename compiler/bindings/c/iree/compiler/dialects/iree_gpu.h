// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECTS_IREE_GPU_H
#define IREE_COMPILER_DIALECTS_IREE_GPU_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// The following C API is **NOT STABLE** and likely to change in the future.
// It mirrors the IREE GPU Dialect which is not stable itself.

MLIR_CAPI_EXPORTED bool
ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID
ireeGPUReorderWorkgroupsStrategyAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUReorderWorkgroupsStrategyAttrGet(MlirContext mlirCtx, uint32_t value);

MLIR_CAPI_EXPORTED uint32_t
ireeGPUReorderWorkgroupsStrategyAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED
bool ireeAttributeIsAGPUPipelineOptionsAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute ireeGPUPipelineOptionsAttrGet(
    MlirContext mlirCtx, bool *prefetchSharedMemory,
    bool *noReduceSharedMemoryBankConflicts, bool *useIgemmConvolution,
    MlirAttribute *reorderWorkgroupsStrategy);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUPipelineOptionsAttrGetPrefetchSharedMemory(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUPipelineOptionsAttrGetNoReduceSharedMemoryBankConflicts(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUPipelineOptionsAttrGetUseIgemmConvolution(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUPipelineOptionsAttrGetReorderWorkgroupsStrategy(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeGPUPipelineOptionsAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool ireeAttributeIsAGPUMMAIntrinsicAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeGPUMMAIntrinsicAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute ireeGPUMMAIntrinsicAttrGet(MlirContext mlirCtx,
                                                            uint32_t value);

MLIR_CAPI_EXPORTED uint32_t ireeGPUMMAIntrinsicAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool ireeAttributeIsAGPUMMAAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeGPUMMAAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute ireeGPUMMAAttrGet(MlirContext mlirCtx,
                                                   uint32_t value);

struct ireeGPUMMAInfo {
  MlirType aElementType;
  MlirType bElementType;
  MlirType cElementType;
  MlirType aVectorType;
  MlirType bVectorType;
  MlirType cVectorType;
  int64_t mElements;
  int64_t nElements;
  int64_t kElements;
};

MLIR_CAPI_EXPORTED ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
ireeAttributeIsAGPULoweringConfigAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeGPULoweringConfigAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute ireeGPULoweringConfigAttrGet(
    MlirContext mlirCtx, MlirAttribute attributesDictionary);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_DIALECTS_IREE_GPU_H
