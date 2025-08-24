// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>
#include <type_traits>
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
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

MlirAttribute ireeGPUComputeBitwidthsAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *mlirCtx = unwrap(ctx);
  return wrap(mlir::iree_compiler::IREE::GPU::ComputeBitwidthsAttr::get(
      mlirCtx,
      static_cast<mlir::iree_compiler::IREE::GPU::ComputeBitwidths>(value)));
}

bool ireeAttributeIsAGPUComputeBitwidthsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::ComputeBitwidthsAttr>(
      unwrap(attr));
}

uint32_t ireeGPUComputeBitwidthsAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUComputeBitwidthsAttr(attr) &&
         "attr is not a GPUComputeBitwidthsAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::ComputeBitwidthsAttr>(
          unwrap(attr))
          .getValue());
}

MlirTypeID ireeGPUComputeBitwidthsAttrGetTypeID() {
  return wrap(
      mlir::iree_compiler::IREE::GPU::ComputeBitwidthsAttr::getTypeID());
}

MlirAttribute ireeGPUStorageBitwidthsAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *mlirCtx = unwrap(ctx);
  return wrap(mlir::iree_compiler::IREE::GPU::StorageBitwidthsAttr::get(
      mlirCtx,
      static_cast<mlir::iree_compiler::IREE::GPU::StorageBitwidths>(value)));
}

uint32_t ireeGPUStorageBitwidthsAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUStorageBitwidthsAttr(attr) &&
         "attr is not a GPUStorageBitwidthsAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::StorageBitwidthsAttr>(
          unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUStorageBitwidthsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::StorageBitwidthsAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUStorageBitwidthsAttrGetTypeID() {
  return wrap(
      mlir::iree_compiler::IREE::GPU::StorageBitwidthsAttr::getTypeID());
}

MlirAttribute ireeGPUSubgroupOpsAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *mlirCtx = unwrap(ctx);
  return wrap(mlir::iree_compiler::IREE::GPU::SubgroupOpsAttr::get(
      mlirCtx,
      static_cast<mlir::iree_compiler::IREE::GPU::SubgroupOps>(value)));
}

uint32_t ireeGPUSubgroupOpsAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUSubgroupOpsAttr(attr) &&
         "attr is not a GPUSubgroupOpsAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::SubgroupOpsAttr>(unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUSubgroupOpsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::SubgroupOpsAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUSubgroupOpsAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::SubgroupOpsAttr::getTypeID());
}

MlirAttribute ireeGPUDotProductOpsAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *mlirCtx = unwrap(ctx);
  return wrap(mlir::iree_compiler::IREE::GPU::DotProductOpsAttr::get(
      mlirCtx,
      static_cast<mlir::iree_compiler::IREE::GPU::DotProductOps>(value)));
}

uint32_t ireeGPUDotProductOpsAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUDotProductOpsAttr(attr) &&
         "attr is not a GPUDotProductOpsAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::DotProductOpsAttr>(
          unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUDotProductOpsAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::DotProductOpsAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUDotProductOpsAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::DotProductOpsAttr::getTypeID());
}

bool ireeAttributeIsAGPUMMAIntrinsicAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
      unwrap(attr));
}

bool ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUMMAIntrinsicAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::getTypeID());
}

MlirTypeID ireeGPUVirtualMMAIntrinsicAttrGetTypeID() {
  return wrap(
      mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::getTypeID());
}

static_assert(
    std::is_same_v<uint32_t, std::underlying_type_t<
                                 mlir::iree_compiler::IREE::GPU::MMAIntrinsic>>,
    "Enum type changed");

static_assert(
    std::is_same_v<uint32_t,
                   std::underlying_type_t<
                       mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>>,
    "Enum type changed");

MlirAttribute ireeGPUMMAIntrinsicAttrGet(MlirContext mlirCtx, uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

MlirAttribute ireeGPUVirtualMMAIntrinsicAttrGet(MlirContext mlirCtx,
                                                uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
      ctx,
      static_cast<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>(value)));
}

uint32_t ireeGPUMMAIntrinsicAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUMMAIntrinsicAttr(attr) &&
         "attr is not a GPUMMAIntrinsicAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(unwrap(attr))
          .getValue());
}

uint32_t ireeGPUVirtualMMAIntrinsicAttrGetValue(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(attr) &&
         "attr is not a GPUVirtualMMAIntrinsicAttr");
  return static_cast<uint32_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr>(
          unwrap(attr))
          .getValue());
}

bool ireeAttributeIsAGPUMMAAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
}

bool ireeAttributeIsAGPUVirtualMMAAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::VirtualMMAAttr>(
      unwrap(attr));
}

MlirTypeID ireeGPUMMAAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAAttr::getTypeID());
}

MlirTypeID ireeGPUVirtualMMAAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::VirtualMMAAttr::getTypeID());
}

MlirAttribute ireeGPUMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

MlirAttribute ireeGPUVirtualMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::VirtualMMAAttr::get(
      ctx,
      static_cast<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>(value)));
}

ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr) {
  return llvm::TypeSwitch<mlir::Attribute, ireeGPUMMAInfo>(unwrap(attr))
      .Case<mlir::iree_compiler::IREE::GPU::MMAAttr,
            mlir::iree_compiler::IREE::GPU::VirtualMMAAttr>([](auto mma) {
        ireeGPUMMAInfo info = {};
        auto [aType, bType, cType] = mma.getABCElementTypes();
        info.aElementType = wrap(aType);
        info.bElementType = wrap(bType);
        info.cElementType = wrap(cType);

        auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
        info.aVectorType = wrap(aVecType);
        info.bVectorType = wrap(bVecType);
        info.cVectorType = wrap(cVecType);

        std::tie(info.mElements, info.nElements, info.kElements) =
            mma.getMNKShape();

        return info;
      })
      .Default([](mlir::Attribute) -> ireeGPUMMAInfo {
        assert(false && "Unexpected attribute type for MMA info");
        return {};
      });
}

MlirAttribute ireeGPUMMAAttrGetVirtualMMAIntrinsic(MlirAttribute attr) {
  assert(ireeAttributeIsAGPUMMAAttr(attr) && "attr is not a MMAAttr");
  auto mma = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
  llvm::SmallVector<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
      virtualIntrinsics = mma.getVirtualIntrinsics();

  llvm::SmallVector<int64_t> rawValues;
  for (auto v : virtualIntrinsics) {
    rawValues.push_back(static_cast<int64_t>(v));
  }

  mlir::MLIRContext *ctx = mma.getContext();
  mlir::Builder builder(ctx);
  return wrap(builder.getI64ArrayAttr(rawValues));
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

ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  ireeGPUTileSizes tilesizes = {};
  mlir::DictionaryAttr dict =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr))
          .getAttributes();

  llvm::StringRef workgroupName =
      mlir::iree_compiler::IREE::GPU::getTilingLevelName(
          mlir::iree_compiler::IREE::GPU::TilingLevel::Workgroup);

  if (auto workgroupArray = dict.getAs<mlir::ArrayAttr>(workgroupName)) {
    tilesizes.workgroupAttr = wrap(workgroupArray);
  }

  llvm::StringRef reductionName =
      mlir::iree_compiler::IREE::GPU::getTilingLevelName(
          mlir::iree_compiler::IREE::GPU::TilingLevel::Reduction);
  if (auto reductionArray = dict.getAs<mlir::ArrayAttr>(reductionName)) {
    tilesizes.reductionAttr = wrap(reductionArray);
  }
  return tilesizes;
}

ireeGPUSubgroupCountInfo
ireeGPULoweringConfigAttrGetSubgroupCount(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  auto loweringConfigAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr));
  std::optional<int64_t> subgroupMCount =
      mlir::iree_compiler::IREE::GPU::getSubgroupMCount(loweringConfigAttr);
  std::optional<int64_t> subgroupNCount =
      mlir::iree_compiler::IREE::GPU::getSubgroupNCount(loweringConfigAttr);

  ireeGPUSubgroupCountInfo info = {};

  if (subgroupMCount) {
    info.subgroupMCountAttr = wrap(mlir::IntegerAttr::get(
        mlir::IndexType::get(loweringConfigAttr.getContext()),
        *subgroupMCount));
  }

  if (subgroupNCount) {
    info.subgroupNCountAttr = wrap(mlir::IntegerAttr::get(
        mlir::IndexType::get(loweringConfigAttr.getContext()),
        *subgroupNCount));
  }
  return info;
}

MlirAttribute ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr) {
  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
  auto loweringConfigAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr));

  mlir::iree_compiler::IREE::Codegen::InnerTileDescAttrInterface mma_attr =
      mlir::iree_compiler::IREE::GPU::getMmaKind(loweringConfigAttr);

  return wrap(mma_attr);
}

ireeGPUMMASingleSubgroupLayout
ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
  assert((ireeAttributeIsAGPUMMAIntrinsicAttr(attr) ||
          ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(attr)) &&
         "Expected MMA or VirtualMMA Intrinsic");

  mlir::Attribute baseAttr = unwrap(attr);
  mlir::iree_compiler::IREE::GPU::MMASingleSubgroupLayout layout;
  mlir::iree_compiler::IREE::GPU::MMAFragment frag =
      static_cast<mlir::iree_compiler::IREE::GPU::MMAFragment>(fragment);

  if (auto intrinsicAttr =
          llvm::dyn_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
              baseAttr)) {
    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
        intrinsicAttr.getValue(), frag);
  } else if (auto virtualIntrinsicAttr = llvm::dyn_cast<
                 mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr>(
                 baseAttr)) {
    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
        virtualIntrinsicAttr.getValue(), frag);
  } else {
    assert(false &&
           "Unreachable: attribute must be MMA or VirtualMMA intrinsic");
  }

  mlir::MLIRContext *context = baseAttr.getContext();
  mlir::Builder builder(context);

  ireeGPUMMASingleSubgroupLayout result = {};
  result.outer = wrap(builder.getI64ArrayAttr(layout.outer));
  result.thread = wrap(builder.getI64ArrayAttr(layout.thread));
  result.tstrides = wrap(builder.getI64ArrayAttr(layout.tstrides));
  result.element = wrap(builder.getI64ArrayAttr(layout.element));
  return result;
}

bool ireeAttributeIsAGPUMMAOpsArrayAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr>(
      unwrap(attr));
}

MlirAttribute ireeGPUMMAOpsArrayAttrGet(MlirContext ctx,
                                        const MlirAttribute *mmaAttrs,
                                        size_t numAttrs) {
  mlir::MLIRContext *mlirCtx = unwrap(ctx);
  std::vector<mlir::iree_compiler::IREE::GPU::MMAAttr> attrs;
  attrs.reserve(numAttrs);
  for (size_t i = 0; i < numAttrs; ++i) {
    auto mmaAttr = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(
        unwrap(mmaAttrs[i]));
    attrs.push_back(mmaAttr);
  }
  auto mmaOpsArrayAttr =
      mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr::get(mlirCtx, attrs);
  return wrap(mmaOpsArrayAttr);
}

MlirAttribute ireeGPUMMAOpsArrayAttrGetValue(MlirAttribute attr) {
  auto mmaOpsArrayAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr>(unwrap(attr));
  auto valueRef = mmaOpsArrayAttr.getValue();

  std::vector<mlir::Attribute> attrs;
  for (const auto &mmaAttr : valueRef) {
    attrs.push_back(mmaAttr);
  }
  return wrap(mlir::ArrayAttr::get(mmaOpsArrayAttr.getContext(), attrs));
}

MlirAttribute ireeGPUMMAOpsArrayAttrGetElement(MlirAttribute attr,
                                               size_t index) {
  auto mmaOpsArrayAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr>(unwrap(attr));
  return wrap(mmaOpsArrayAttr[index]);
}

size_t ireeGPUMMAOpsArrayAttrGetSize(MlirAttribute attr) {
  auto mmaOpsArrayAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr>(unwrap(attr));
  return mmaOpsArrayAttr.size();
}

MlirTypeID ireeGPUMMAOpsArrayAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr::getTypeID());
}

bool ireeAttributeIsAGPUTargetWgpAttr(MlirAttribute attr) {
  return llvm::isa<mlir::iree_compiler::IREE::GPU::TargetWgpAttr>(unwrap(attr));
}

MlirTypeID ireeGPUTargetWgpAttrGetTypeID() {
  return wrap(mlir::iree_compiler::IREE::GPU::TargetWgpAttr::getTypeID());
}

MlirAttribute ireeGPUTargetWgpAttrGet(MlirContext mlirCtx,
                                      ireeGPUTargetWgpInfo targetInfo) {
  assert(!mlirAttributeIsNull(targetInfo.compute) &&
         ireeAttributeIsAGPUComputeBitwidthsAttr(targetInfo.compute) &&
         "Invalid compute bitwidths attr");

  assert(!mlirAttributeIsNull(targetInfo.storage) &&
         ireeAttributeIsAGPUStorageBitwidthsAttr(targetInfo.storage) &&
         "Invalid storage bitwidths attr");

  assert(!mlirAttributeIsNull(targetInfo.subgroup) &&
         ireeAttributeIsAGPUSubgroupOpsAttr(targetInfo.subgroup) &&
         "Invalid subgroup ops attr");

  assert(!mlirAttributeIsNull(targetInfo.dot) &&
         ireeAttributeIsAGPUDotProductOpsAttr(targetInfo.dot) &&
         "Invalid dot product ops attr");

  assert(ireeAttributeIsAGPUMMAOpsArrayAttr(targetInfo.mma) &&
         "Invalid MMA ops array attr");

  assert(!mlirAttributeIsNull(targetInfo.subgroup_size_choices) &&
         mlirAttributeIsADenseI32Array(targetInfo.subgroup_size_choices) &&
         "Invalid subgroup size choices attr");

  assert(!mlirAttributeIsNull(targetInfo.max_workgroup_sizes) &&
         mlirAttributeIsADenseI32Array(targetInfo.max_workgroup_sizes) &&
         "Invalid max workgroup sizes attr");

  assert(!mlirAttributeIsNull(targetInfo.max_workgroup_counts) &&
         mlirAttributeIsADenseI32Array(targetInfo.max_workgroup_counts) &&
         "Invalid max workgroup counts attr");

  auto compute =
      llvm::cast<mlir::iree_compiler::IREE::GPU::ComputeBitwidthsAttr>(
          unwrap(targetInfo.compute));
  auto storage =
      llvm::cast<mlir::iree_compiler::IREE::GPU::StorageBitwidthsAttr>(
          unwrap(targetInfo.storage));
  auto subgroup = llvm::cast<mlir::iree_compiler::IREE::GPU::SubgroupOpsAttr>(
      unwrap(targetInfo.subgroup));
  auto dot = llvm::cast<mlir::iree_compiler::IREE::GPU::DotProductOpsAttr>(
      unwrap(targetInfo.dot));
  auto mma = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr>(
      unwrap(targetInfo.mma));
  auto subgroup_size_choices = llvm::cast<mlir::DenseI32ArrayAttr>(
      unwrap(targetInfo.subgroup_size_choices));
  auto max_workgroup_sizes = llvm::cast<mlir::DenseI32ArrayAttr>(
      unwrap(targetInfo.max_workgroup_sizes));
  auto max_workgroup_counts = llvm::cast<mlir::DenseI32ArrayAttr>(
      unwrap(targetInfo.max_workgroup_counts));

  std::optional<int32_t> max_load_instruction_bits;
  if (targetInfo.max_load_instruction_bits != -1) {
    max_load_instruction_bits = targetInfo.max_load_instruction_bits;
  }

  std::optional<int32_t> simds_per_wgp;
  if (targetInfo.simds_per_wgp != -1) {
    simds_per_wgp = targetInfo.simds_per_wgp;
  }

  std::optional<int32_t> vgpr_space_bits;
  if (targetInfo.vgpr_space_bits != -1) {
    vgpr_space_bits = targetInfo.vgpr_space_bits;
  }

  auto extra =
      llvm::cast_if_present<mlir::DictionaryAttr>(unwrap(targetInfo.extra));

  // Create null scaled_mma attribute
  auto scaled_mma =
      llvm::cast<mlir::iree_compiler::IREE::GPU::ScaledMMAOpsArrayAttr>(
          unwrap(mlirAttributeGetNull()));
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::TargetWgpAttr::get(
      ctx, compute, storage, subgroup, dot, mma, scaled_mma,
      subgroup_size_choices, max_workgroup_sizes,
      targetInfo.max_thread_count_per_workgroup,
      targetInfo.max_workgroup_memory_bytes, max_workgroup_counts,
      max_load_instruction_bits, simds_per_wgp, vgpr_space_bits, extra));
}

ireeGPUTargetWgpInfo ireeGPUTargetWgpAttrGetInfo(MlirAttribute attr) {
  auto targetWgpAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::TargetWgpAttr>(unwrap(attr));

  ireeGPUTargetWgpInfo targetInfo = {};
  targetInfo.compute = wrap(targetWgpAttr.getCompute());
  targetInfo.storage = wrap(targetWgpAttr.getStorage());
  targetInfo.subgroup = wrap(targetWgpAttr.getSubgroup());
  targetInfo.dot = wrap(targetWgpAttr.getDot());
  targetInfo.mma = wrap(targetWgpAttr.getMma());
  targetInfo.subgroup_size_choices =
      wrap(targetWgpAttr.getSubgroupSizeChoices());
  targetInfo.max_workgroup_sizes = wrap(targetWgpAttr.getMaxWorkgroupSizes());
  targetInfo.max_thread_count_per_workgroup =
      targetWgpAttr.getMaxThreadCountPerWorkgroup();
  targetInfo.max_workgroup_memory_bytes =
      targetWgpAttr.getMaxWorkgroupMemoryBytes();
  targetInfo.max_workgroup_counts = wrap(targetWgpAttr.getMaxWorkgroupCounts());

  targetInfo.max_load_instruction_bits =
      targetWgpAttr.getMaxLoadInstructionBits().value_or(-1);

  targetInfo.simds_per_wgp = targetWgpAttr.getSimdsPerWgp().value_or(-1);
  targetInfo.vgpr_space_bits = targetWgpAttr.getVgprSpaceBits().value_or(-1);
  targetInfo.extra = wrap(targetWgpAttr.getExtra());
  return targetInfo;
}
