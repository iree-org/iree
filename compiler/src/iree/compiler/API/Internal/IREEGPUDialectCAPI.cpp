// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/dialects/iree_gpu.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

// Macro to check that enum values match across C++ and C API.
#define ASSERT_ENUM_VALS_MATCH(CppEnumName, CAPIEnumName, ValueName)           \
  static_assert(llvm::to_underlying(CppEnumName ::ValueName) ==                \
                    llvm::to_underlying(CAPIEnumName##ValueName),              \
                #CppEnumName " and " #CAPIEnumName " have diverged")

bool ireeAttributeIsAGPUPipelineOptionsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr>(
      unwrap(attr));
}

MlirAttribute
ireeGPUPipelineOptionsAttrGet(MlirContext mlirCtx, bool *prefetchSharedMemory,
                              bool *noReduceSharedMemoryBankConflicts,
                              bool *useIgemmConvolution,
                              MlirAttribute *reorderWorkgroupsStrategy) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  mlir::Builder b(ctx);
  auto prefetchSharedMemoryAttr = mlir::BoolAttr();
  if (prefetchSharedMemory) {
    prefetchSharedMemoryAttr = b.getBoolAttr(*prefetchSharedMemory);
  }
  auto noReduceSharedMemoryBankConflictsAttr = mlir::BoolAttr();
  if (noReduceSharedMemoryBankConflicts) {
    noReduceSharedMemoryBankConflictsAttr =
        b.getBoolAttr(*noReduceSharedMemoryBankConflicts);
  }
  auto useIgemmConvolutionAttr = mlir::BoolAttr();
  if (useIgemmConvolution) {
    useIgemmConvolutionAttr = b.getBoolAttr(*useIgemmConvolution);
  }
  auto strategyAttr =
      mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr();
  if (reorderWorkgroupsStrategy) {
    strategyAttr = llvm::dyn_cast<
        mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr>(
        unwrap(*reorderWorkgroupsStrategy));
  }
  return wrap(mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr::get(
      ctx, prefetchSharedMemoryAttr, noReduceSharedMemoryBankConflictsAttr,
      useIgemmConvolutionAttr, strategyAttr));
}

MlirAttribute
ireeGPUPipelineOptionsAttrGetPrefetchSharedMemory(MlirAttribute attr) {
  auto gpuAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr>(
          unwrap(attr));
  return wrap(gpuAttr.getPrefetchSharedMemory());
}

MlirAttribute ireeGPUPipelineOptionsAttrGetNoReduceSharedMemoryBankConflicts(
    MlirAttribute attr) {
  auto gpuAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr>(
          unwrap(attr));
  return wrap(gpuAttr.getNoReduceSharedMemoryBankConflicts());
}

MlirAttribute
ireeGPUPipelineOptionsAttrGetUseIgemmConvolution(MlirAttribute attr) {
  auto gpuAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr>(
          unwrap(attr));
  return wrap(gpuAttr.getUseIgemmConvolution());
}

MlirAttribute
ireeGPUPipelineOptionsAttrGetReorderWorkgroupsStrategy(MlirAttribute attr) {
  auto gpuAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr>(
          unwrap(attr));
  return wrap(gpuAttr.getReorderWorkgroupsStrategy());
}

MlirTypeID ireeGPUPipelineOptionsAttrGetTypeID() {
  return wrap(
      mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr::getTypeID());
}

#define X_DO(EnumName, EnumValue)                                              \
  ASSERT_ENUM_VALS_MATCH(                                                      \
      mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategy,               \
      ireeGPUReorderWorkgroupsStrategyEnum, EnumName);

IREE_GPU_FOR_ALL_REORDER_WORKGROUP_VALUES

#undef X_DO

#undef FOR_ALL_REORDER_WORKGROUP_VALUES

bool ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr(MlirAttribute attr) {
  return llvm::isa<
      mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUReorderWorkgroupsStrategyAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr::
                  getTypeID());
}

MlirAttribute ireeGPUReorderWorkgroupsStrategyAttrGet(
    MlirContext mlirCtx, ireeGPUReorderWorkgroupsStrategyEnum value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(
      mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr::get(
          ctx, static_cast<
                   mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategy>(
                   value)));
}

ireeGPUReorderWorkgroupsStrategyEnum
ireeGPUReorderWorkgroupsStrategyAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr(attr) &&
         "attr is not a GPUReorderWorkgroupsStrategyAttr");
  return static_cast<ireeGPUReorderWorkgroupsStrategyEnum>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr>(
          unwrap(attr))
          .getValue());
}

#define X_DO(EnumName, EnumVal)                                                \
  ASSERT_ENUM_VALS_MATCH(mlir::iree_compiler::IREE::GPU::MMAIntrinsic,         \
                         ireeGPUMMAIntrinsicEnum, EnumName);

IREE_GPU_FOR_ALL_MMA_INTRINSIC_VALUES

#undef X_DO

bool ireeAttributeIsAGPUMMAIntrinsicAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUMMAIntrinsicAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::getTypeID());
}

MlirAttribute ireeGPUMMAIntrinsicAttrGet(MlirContext mlirCtx,
                                         ireeGPUMMAIntrinsicEnum value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

ireeGPUMMAIntrinsicEnum ireeGPUMMAIntrinsicAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUMMAIntrinsicAttr(attr) &&
         "attr is not a GPUMMAIntrinsicAttr");
  return static_cast<ireeGPUMMAIntrinsicEnum>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUMMAAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
}

MlirTypeID ireeGPUMMAAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAAttr::getTypeID());
}

MlirAttribute ireeGPUMMAAttrGet(MlirContext mlirCtx,
                                ireeGPUMMAIntrinsicEnum value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUMMAAttr(attr) && "attr is not a MMAAttr");
  auto mma = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));

  ireeGPUMMAInfo info = {};
  auto [aType, bType, cType] = mma.getABCElementTypes();
  info.aElementType = wrap(aType);
  info.bElementType = wrap(bType);
  info.cElementType = wrap(cType);

  auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
  info.aVectorType = wrap(aVecType);
  info.bVectorType = wrap(bVecType);
  info.cVectorType = wrap(cVecType);

  std::tie(info.mElements, info.nElements, info.kElements) = mma.getMNKShape();
  return info;
}

#undef ASSERT_ENUM_VALS_MATCH
