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
