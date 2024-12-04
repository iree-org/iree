// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>
#include <type_traits>
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/dialects/iree_gpu.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinAttributes.h"

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

bool ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr(MlirAttribute attr) {
  return llvm::isa<
      mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUReorderWorkgroupsStrategyAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr::
                  getTypeID());
}

static_assert(
    std::is_same_v<
        uint32_t,
        std::underlying_type_t<
            mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategy>>,
    "Enum type changed");

MlirAttribute ireeGPUReorderWorkgroupsStrategyAttrGet(MlirContext mlirCtx,
                                                      uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(
      mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr::get(
          ctx, static_cast<
                   mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategy>(
                   value)));
}

uint32_t ireeGPUReorderWorkgroupsStrategyAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr(attr) &&
         "attr is not a GPUReorderWorkgroupsStrategyAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::ReorderWorkgroupsStrategyAttr>(
          unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUMMAIntrinsicAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUMMAIntrinsicAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::getTypeID());
}

static_assert(
    std::is_same_v<uint32_t, std::underlying_type_t<
                                 mlir::iree_compiler::IREE::GPU::MMAIntrinsic>>,
    "Enum type changed");

MlirAttribute ireeGPUMMAIntrinsicAttrGet(MlirContext mlirCtx, uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

uint32_t ireeGPUMMAIntrinsicAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUMMAIntrinsicAttr(attr) &&
         "attr is not a GPUMMAIntrinsicAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUMMAAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
}

MlirTypeID ireeGPUMMAAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAAttr::getTypeID());
}

MlirAttribute ireeGPUMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
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

bool ireeAttributeIsAGPULoweringConfigAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPULoweringConfigAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::LoweringConfigAttr::getTypeID());
}

MlirAttribute ireeGPULoweringConfigAttrGet(MlirContext mlirCtx,
                                           MlirAttribute attributesDictionary) {
  assert(mlirAttributeIsADictionary(attributesDictionary));
  auto attributes =
      llvm::cast<mlir::DictionaryAttr>(unwrap(attributesDictionary));
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(
      mlir::iree_compiler::IREE::GPU::LoweringConfigAttr::get(ctx, attributes));
}

MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  return wrap(llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
                  unwrap(attr))
                  .getAttributes());
}

void ireeGPULoweringConfigAttrGetWorkgroupTileSizes(
    MlirAttribute attr, size_t *len, int64_t *workgroupTileSizes) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  mlir::DictionaryAttr dict =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr))
          .getAttributes();
  constexpr mlir::StringLiteral workgroupName = "workgroup";
  mlir::ArrayAttr array = dict.getAs<mlir::ArrayAttr>(workgroupName);

  if (!array ||
      !llvm::all_of(array.getValue(), llvm::IsaPred<mlir::IntegerAttr>)) {
    *len = -1;
    return;
  }

  if (!workgroupTileSizes) {
    *len = array.size();
    return;
  }

  assert(*len == array.size() &&
         "the length should match the number of tilesizes in the array");

  for (size_t i = 0, e = array.size(); i < e; ++i) {
    mlir::IntegerAttr element =
        llvm::cast<mlir::IntegerAttr>(array.getValue()[i]);
    workgroupTileSizes[i] = element.getInt();
  }
  return;
}

void ireeGPULoweringConfigAttrGetReductionTileSizes(
    MlirAttribute attr, size_t *len, int64_t *reductionTileSizes) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  mlir::DictionaryAttr dict =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr))
          .getAttributes();
  constexpr mlir::StringLiteral reductionName = "reduction";
  mlir::ArrayAttr array = dict.getAs<mlir::ArrayAttr>(reductionName);

  if (!array ||
      !llvm::all_of(array.getValue(), llvm::IsaPred<mlir::IntegerAttr>)) {
    *len = -1;
    return;
  }

  if (!reductionTileSizes) {
    *len = array.size();
    return;
  }

  assert(*len == array.size() &&
         "the length should match the number of tilesizes in the array");

  for (size_t i = 0, e = array.size(); i < e; ++i) {
    mlir::IntegerAttr element =
        llvm::cast<mlir::IntegerAttr>(array.getValue()[i]);
    reductionTileSizes[i] = element.getInt();
  }
  return;
}

MlirAttribute ireeGPULoweringConfigAttrGetSubgroupMCount(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  mlir::DictionaryAttr dict =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr))
          .getAttributes();

  constexpr mlir::StringLiteral kSubgroupMCountName = "subgroup_m_count";
  mlir::IntegerAttr subgroup_m_count_attr =
      dict.getAs<mlir::IntegerAttr>(kSubgroupMCountName);

  return wrap(subgroup_m_count_attr);
}

MlirAttribute ireeGPULoweringConfigAttrGetSubgroupNCount(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  mlir::DictionaryAttr dict =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr))
          .getAttributes();

  constexpr mlir::StringLiteral kSubgroupNCountName = "subgroup_n_count";
  mlir::IntegerAttr subgroup_n_count_attr =
      dict.getAs<mlir::IntegerAttr>(kSubgroupNCountName);

  return wrap(subgroup_n_count_attr);
}

MlirAttribute ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  mlir::DictionaryAttr dict =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr))
          .getAttributes();

  constexpr mlir::StringLiteral kMmaKindName = "mma_kind";
  mlir::iree_compiler::IREE::GPU::MmaInterfaceAttr mma_attr =
      dict.getAs<mlir::iree_compiler::IREE::GPU::MmaInterfaceAttr>(
          kMmaKindName);

  return wrap(mma_attr);
}
