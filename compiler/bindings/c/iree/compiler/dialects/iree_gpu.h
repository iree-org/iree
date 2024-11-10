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

#define IREE_GPU_FOR_ALL_REORDER_WORKGROUP_VALUES                              \
  X_DO(None, 0)                                                                \
  X_DO(Transpose, 1)

enum ireeGPUReorderWorkgroupsStrategyEnum {
#define X_DO(EnumName, EnumValue)                                              \
  ireeGPUReorderWorkgroupsStrategyEnum##EnumName = EnumValue,

  IREE_GPU_FOR_ALL_REORDER_WORKGROUP_VALUES

#undef X_DO
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

#define IREE_GPU_FOR_ALL_MMA_INTRINSIC_VALUES                                  \
  X_DO(MFMA_F32_16x16x4_F32, 0x0900)                                           \
  X_DO(MFMA_F32_16x16x16_F16, 0x0910)                                          \
  X_DO(MFMA_F32_32x32x8_F16, 0x0911)                                           \
  X_DO(MFMA_F32_16x16x16_BF16, 0x0920)                                         \
  X_DO(MFMA_F32_32x32x8_BF16, 0x0921)                                          \
  X_DO(MFMA_F32_16x16x32_F8E4M3FNUZ, 0x0940)                                   \
  X_DO(MFMA_F32_16x16x32_F8E5M2FNUZ, 0x0930)                                   \
  X_DO(MFMA_I32_16x16x32_I8, 0x0980)                                           \
  X_DO(MFMA_I32_32x32x16_I8, 0x0981)                                           \
  X_DO(MFMA_I32_16x16x16_I8, 0x0880)                                           \
  X_DO(MFMA_I32_32x32x8_I8, 0x0881)                                            \
  X_DO(WMMA_F32_16x16x16_F16, 0x0010)                                          \
  X_DO(WMMA_F16_16x16x16_F16, 0x0011)                                          \
  X_DO(WMMA_I32_16x16x16_I8, 0x0080)

enum ireeGPUMMAIntrinsicEnum {
#define X_DO(EnumName, EnumValue) ireeGPUMMAIntrinsicEnum##EnumName = EnumValue,

  IREE_GPU_FOR_ALL_MMA_INTRINSIC_VALUES

#undef X_DO
};

MLIR_CAPI_EXPORTED bool ireeAttributeIsAGPUMMAIntrinsicAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeGPUMMAIntrinsicAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUMMAIntrinsicAttrGet(MlirContext mlirCtx, ireeGPUMMAIntrinsicEnum value);

MLIR_CAPI_EXPORTED ireeGPUMMAIntrinsicEnum
ireeGPUMMAIntrinsicAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool ireeAttributeIsAGPUMMAAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID ireeGPUMMAAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute
ireeGPUMMAAttrGet(MlirContext mlirCtx, ireeGPUMMAIntrinsicEnum value);

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

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_DIALECTS_IREE_GPU_H
