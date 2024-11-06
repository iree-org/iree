// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/dialects/iree_gpu.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

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

static_assert(
    static_cast<uint32_t>(ireeGPUReorderWorkgroupsStrategyEnumNone) ==
            static_cast<uint32_t>(mlir::iree_compiler::IREE::GPU::
                                      ReorderWorkgroupsStrategy::None) &&
        static_cast<uint32_t>(ireeGPUReorderWorkgroupsStrategyEnumTranspose) ==
            static_cast<uint32_t>(mlir::iree_compiler::IREE::GPU::
                                      ReorderWorkgroupsStrategy::Transpose) &&
        static_cast<uint32_t>(ireeGPUReorderWorkgroupsStrategyEnumTranspose) ==
            mlir::iree_compiler::IREE::GPU::
                getMaxEnumValForReorderWorkgroupsStrategy(),
    "ireeGPUReorderWorkgroupsStrategyEnum and "
    "mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategy definitions "
    "have diverged");

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
