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

enum ireeGPUReorderWorkgroupsStrategyEnum {
  ireeGPUReorderWorkgroupsStrategyEnumNone = 0,
  ireeGPUReorderWorkgroupsStrategyEnumSwizzle = 1,
  ireeGPUReorderWorkgroupsStrategyEnumTranspose = 2,
};

MLIR_CAPI_EXPORTED bool
ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID
ireeGPUReorderWorkgroupsStrategyAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute ireeGPUReorderWorkgroupsStrategyAttrGet(
    MlirContext mlirCtx, ireeGPUReorderWorkgroupsStrategyEnum value);

MLIR_CAPI_EXPORTED ireeGPUReorderWorkgroupsStrategyEnum
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

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_DIALECTS_IREE_GPU_H
