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
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
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
ireeGPUPipelineOptionsAttrGet(MlirContext mlirCtx, int64_t *prefetchNumStages,
                              bool *noReduceSharedMemoryBankConflicts,
                              bool *useIgemmConvolution,
                              MlirAttribute *reorderWorkgroupsStrategy) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  mlir::Builder b(ctx);
  std::optional<int64_t> prefetchNumStagesOpt;
  if (prefetchNumStages) {
    prefetchNumStagesOpt = *prefetchNumStages;
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
      ctx, prefetchNumStagesOpt, noReduceSharedMemoryBankConflictsAttr,
      useIgemmConvolutionAttr, strategyAttr));
}

MlirAttribute
ireeGPUPipelineOptionsAttrGetPrefetchNumStages(MlirAttribute attr) {
  auto gpuAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::GPUPipelineOptionsAttr>(
          unwrap(attr));
  std::optional<int64_t> value = gpuAttr.getPrefetchNumStages();
  if (!value) {
    return {nullptr};
  }
  mlir::Builder b(unwrap(attr).getContext());
  return wrap(b.getI64IntegerAttr(*value));
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
    std::is_same_v<
        mma_intrinsic_enum_t,
        std::underlying_type_t<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>>,
    "MMAIntrinsic Enum type changed");

static_assert(
    std::is_same_v<mma_intrinsic_enum_t,
                   std::underlying_type_t<
                       mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>>,
    "VirtualMMAIntrinsic Enum type changed");

MlirAttribute ireeGPUMMAIntrinsicAttrGet(MlirContext mlirCtx,
                                         mma_intrinsic_enum_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

MlirAttribute ireeGPUVirtualMMAIntrinsicAttrGet(MlirContext mlirCtx,
                                                mma_intrinsic_enum_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
      ctx,
      static_cast<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>(value)));
}

mma_intrinsic_enum_t ireeGPUMMAIntrinsicAttrGetValue(MlirAttribute attr) {
  return static_cast<mma_intrinsic_enum_t>(
      llvm::cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(unwrap(attr))
          .getValue());
}

mma_intrinsic_enum_t
ireeGPUVirtualMMAIntrinsicAttrGetValue(MlirAttribute attr) {
  return static_cast<mma_intrinsic_enum_t>(
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

MlirAttribute ireeGPUMMAAttrGet(MlirContext mlirCtx,
                                mma_intrinsic_enum_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(mlir::iree_compiler::IREE::GPU::MMAAttr::get(
      ctx, static_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>(value)));
}

MlirAttribute ireeGPUVirtualMMAAttrGet(MlirContext mlirCtx,
                                       mma_intrinsic_enum_t value) {
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
  auto mma = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
  llvm::SmallVector<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
      virtualIntrinsics = mma.getVirtualIntrinsics();

  auto rawValues =
      llvm::map_to_vector(virtualIntrinsics, llvm::StaticCastTo<int64_t>);
  mlir::Builder builder(mma.getContext());
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
  auto attributes =
      llvm::cast<mlir::DictionaryAttr>(unwrap(attributesDictionary));
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(
      mlir::iree_compiler::IREE::GPU::LoweringConfigAttr::get(ctx, attributes));
}

MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
                  unwrap(attr))
                  .getAttributes());
}

ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
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

MlirAttribute ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr) {
  auto loweringConfigAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr));

  mlir::iree_compiler::IREE::Codegen::InnerTileDescAttrInterface mma_attr =
      mlir::iree_compiler::IREE::GPU::getMmaKind(loweringConfigAttr);

  return wrap(mma_attr);
}

ireeGPUMMASingleSubgroupLayout
ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t operandIndex) {
  assert((ireeAttributeIsAGPUMMAIntrinsicAttr(attr) ||
          ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(attr)) &&
         "Expected MMA or VirtualMMA Intrinsic");

  mlir::Attribute baseAttr = unwrap(attr);
  mlir::iree_compiler::IREE::GPU::MMASingleSubgroupLayout layout;

  if (auto intrinsicAttr =
          llvm::dyn_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
              baseAttr)) {
    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
        intrinsicAttr.getValue(), operandIndex);
  } else if (auto virtualIntrinsicAttr = llvm::dyn_cast<
                 mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr>(
                 baseAttr)) {
    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
        virtualIntrinsicAttr.getValue(), operandIndex);
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

ireeGPUSubgroupBasisInfo
ireeGPULoweringConfigAttrGetSubgroupBasis(MlirAttribute attr) {
  auto loweringConfigAttr =
      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
          unwrap(attr));

  mlir::FailureOr<mlir::iree_compiler::IREE::GPU::Basis> basisResult =
      mlir::iree_compiler::IREE::GPU::getBasis(
          loweringConfigAttr,
          mlir::iree_compiler::IREE::GPU::TilingLevel::Subgroup);

  ireeGPUSubgroupBasisInfo info = {};
  if (failed(basisResult)) {
    return info;
  }

  mlir::iree_compiler::IREE::GPU::Basis basis = *basisResult;
  mlir::Builder builder(loweringConfigAttr.getContext());
  mlir::ArrayAttr countsAttr = builder.getI64ArrayAttr(basis.counts);
  mlir::ArrayAttr mappingAttr = builder.getI64ArrayAttr(basis.mapping);

  info.countsAttr = wrap(countsAttr);
  info.mappingAttr = wrap(mappingAttr);

  return info;
}

ireeGPUTargetInfo
ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr) {
  auto executableTargetAttr =
      llvm::cast<mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr>(
          unwrap(attr));

  ireeGPUTargetInfo targetInfo = {};
  mlir::MLIRContext *context = executableTargetAttr.getContext();
  mlir::iree_compiler::IREE::GPU::TargetAttr gpuTargetAttr =
      mlir::iree_compiler::getGPUTargetAttr(context, executableTargetAttr);

  if (!gpuTargetAttr) {
    return targetInfo;
  }

  targetInfo.arch =
      wrap(mlir::StringAttr::get(context, gpuTargetAttr.getArch()));
  mlir::iree_compiler::IREE::GPU::TargetWgpAttr wgpAttr =
      gpuTargetAttr.getWgp();

  mlir::Builder builder(context);
  targetInfo.subgroupSizeChoices =
      wrap(builder.getI32ArrayAttr(wgpAttr.getSubgroupSizeChoices()));
  targetInfo.maxWorkgroupSizes =
      wrap(builder.getI32ArrayAttr(wgpAttr.getMaxWorkgroupSizes()));

  targetInfo.maxThreadCountPerWorkgroup =
      wgpAttr.getMaxThreadCountPerWorkgroup();
  targetInfo.maxWorkgroupMemoryBytes = wgpAttr.getMaxWorkgroupMemoryBytes();
  targetInfo.simdsPerWgp = wgpAttr.getSimdsPerWgp().value_or(0);

  if (mlir::iree_compiler::IREE::GPU::TargetChipAttr chipAttr =
          gpuTargetAttr.getChip()) {
    targetInfo.wgpCount = chipAttr.getWgpCount();
  }

  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr({}));
  mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr mmaOpsArray =
      wgpAttr.getMma();
  if (!mmaOpsArray) {
    return targetInfo;
  }

  std::vector<mlir::Attribute> mmaIntrinsicAttrs;
  for (mlir::iree_compiler::IREE::GPU::MMAAttr mmaAttr : mmaOpsArray) {
    mlir::iree_compiler::IREE::GPU::MMAIntrinsic intrinsic =
        mmaAttr.getIntrinsic();
    mmaIntrinsicAttrs.push_back(
        mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(context,
                                                              intrinsic));

    for (mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mmaAttr.getVirtualIntrinsics()) {
      mmaIntrinsicAttrs.push_back(
          mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
              context, virtualIntrinsic));
    }
  }
  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
  return targetInfo;
}

ireeGPUTargetInfo ireeGPUTargetInfoGet(
    MlirContext mlirCtx, const char *arch, const int32_t *subgroupChoices,
    size_t numSubgroupChoices, const int32_t *workgroupSizes,
    size_t numWorkgroupSizes, int32_t threadCount, int32_t memoryBytes,
    uint32_t wgpCount, int32_t simdsPerWgp,
    const mma_intrinsic_enum_t *mmaIntrinsics, size_t numMmaIntrinsics) {
  assert(!mlirContextIsNull(mlirCtx) && "mlirCtx cannot be null");
  assert(arch && "arch cannot be null");

  mlir::MLIRContext *context = unwrap(mlirCtx);
  mlir::Builder builder(context);

  ireeGPUTargetInfo targetInfo = {};

  targetInfo.arch = wrap(mlir::StringAttr::get(context, arch));
  std::vector<int32_t> subgroupChoicesVec(subgroupChoices,
                                          subgroupChoices + numSubgroupChoices);
  targetInfo.subgroupSizeChoices =
      wrap(builder.getI32ArrayAttr(subgroupChoicesVec));
  std::vector<int32_t> workgroupSizesVec(workgroupSizes,
                                         workgroupSizes + numWorkgroupSizes);
  targetInfo.maxWorkgroupSizes =
      wrap(builder.getI32ArrayAttr(workgroupSizesVec));

  targetInfo.maxThreadCountPerWorkgroup = threadCount;
  targetInfo.maxWorkgroupMemoryBytes = memoryBytes;
  targetInfo.wgpCount = wgpCount;
  targetInfo.simdsPerWgp = simdsPerWgp;

  std::vector<mlir::Attribute> mmaIntrinsicAttrs;
  mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
  for (size_t i = 0; i < numMmaIntrinsics; ++i) {
    mma_intrinsic_enum_t enumValue = mmaIntrinsics[i];

    std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
        mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
    if (mmaIntrinsic) {
      auto mmaIntrinsicAttr =
          mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(context,
                                                                *mmaIntrinsic);
      mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
      continue;
    }

    std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
        virtualMmaIntrinsic =
            mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
                enumValue);
    if (virtualMmaIntrinsic) {
      auto virtualMmaIntrinsicAttr =
          mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
              context, *virtualMmaIntrinsic);
      mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
      continue;
    }

    assert(false && "Invalid MMA intrinsic value");
  }
  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));

  return targetInfo;
}

void ireeGPUTargetInfoGetMMAIntrinsics(MlirAttribute mmaIntrinsics,
                                       mma_intrinsic_enum_t *mmaIntrinsicVals,
                                       uint8_t *virtualMmaIntrinsicTags) {
  assert(mlirAttributeIsAArray(mmaIntrinsics) &&
         "mmaIntrinsics must be an array attribute");
  size_t numElements = mlirArrayAttrGetNumElements(mmaIntrinsics);

  for (size_t i = 0; i < numElements; ++i) {
    MlirAttribute element = mlirArrayAttrGetElement(mmaIntrinsics, i);
    if (ireeAttributeIsAGPUMMAIntrinsicAttr(element)) {
      mmaIntrinsicVals[i] = ireeGPUMMAIntrinsicAttrGetValue(element);
      virtualMmaIntrinsicTags[i] = 0; // false.
      continue;
    }

    if (ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(element)) {
      mmaIntrinsicVals[i] = ireeGPUVirtualMMAIntrinsicAttrGetValue(element);
      virtualMmaIntrinsicTags[i] = 1; // true.
      continue;
    }
    assert(false && "Unexpected attribute type in MMA intrinsics array");
  }
}
