// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- GPUEncodingExternalModels.cpp --------------------------------------===//
//
// This file implements the following interfaces for GPU backends:
//
// - IREE::Encoding::LayoutResolverAttr
// - IREE::Encoding::SerializableAttr
// - IREE::Encoding::LayoutMaterializerAttr
// - IREE::Codegen::PackedLayoutMaterializerAttr
//
// Different from CPU backends, we do not transpose narrow-N to narrow-M for a
// combination of reasons:
//
//   1. As linalg.matmul materializes into iree_codegen.inner_tiled, which
//      inherits its semantics from the wrapped intrinsic, we can't rely on any
//      kind of LHS<->RHS symmetry.
//   2. We do not currently use ukernels, which would be one of the main areas
//      to benefit from transposeNarrowN.
//   3. Heuristics for cache-friendly dispatch tiling are internal to the GPU
//      runtime, so we don't need a simplification at that level either.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/GPUEncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <numeric>

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler::IREE::GPU {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;

namespace {

static MMAAttr chooseIntrinsicMMAAttr(TypeRange eTypes, TargetWgpAttr wgp) {
  MMAAttr candidateMma;
  for (MMAAttr mma : wgp.getMma()) {
    // Filter out intrinsics that don't match the element types of this matmul.
    auto [et0, et1, et2] = mma.getABCElementTypes();
    if (et0 != eTypes[0] || et1 != eTypes[1] || et2 != eTypes[2]) {
      continue;
    }
    // If multiple intrinsics are available for the given element types, we have
    // to make a choice. On CDNA3, there may be an intrinsic with larger M/N and
    // smaller K, which would optimize power, and an intrinsic with larger K,
    // which would optimize performance when power is not the bottleneck.
    // Currently we just choose the intrinsic maximizing K, but that can be
    // revisited later.
    if (candidateMma &&
        getKSize(candidateMma.getIntrinsic()) > getKSize(mma.getIntrinsic())) {
      continue;
    }
    candidateMma = mma;
  }
  return candidateMma;
}

static DataTiledMMAInterfaceAttr
chooseDataTiledMMAAttr(TypeRange eTypes, TargetAttr target,
                       IREE::Encoding::EncodingAttr encoding,
                       GPUEncodingResolverAttr resolver) {
  if (!target) {
    return {};
  }
  // First try finding a data-tiled MMA layout through the ukernel provider if
  // one can be found in the config. The ukernel provider is only available in
  // case ukernels are enabled, so we're sure we want to override the default
  // logic.
  DictionaryAttr config = resolver.getConfiguration();
  if (IREE::Codegen::UKernelProviderInterface provider =
          getUKernelProviderFromTarget(config)) {
    auto mma = dyn_cast_if_present<IREE::GPU::DataTiledMMAAttr>(
        provider.getDataLayoutForUKernel(encoding, config));
    if (mma) {
      return mma;
    }
  }

  MLIRContext *ctx = target.getContext();
  IREE::GPU::TargetWgpAttr wgp = target.getWgp();
  if (!wgp.getMaxLoadInstructionBits() || !wgp.getVgprSpaceBits() ||
      !wgp.getSimdsPerWgp()) {
    // Missing workgroup parameters: data tiling not supported on this target.
    return {};
  }

  //
  // Step 1: select a MMAIntrinsic.
  //
  MMAAttr intrinsicMma = chooseIntrinsicMMAAttr(eTypes, wgp);
  if (!intrinsicMma) {
    return {};
  }

  //
  // Step 2: Select the unrolling factors for the generic case where there is no
  //         narrow dimension.
  //

  auto sizeInBits = [](VectorType type) -> int {
    return type.getElementTypeBitWidth() * type.getNumElements();
  };

  auto [intrinsicA, intrinsicB, intrinsicC] = intrinsicMma.getABCVectorTypes();
  // The intrinsicsK factor serves to allow loads from the A and B matrices to
  // use the target ISA's vector loads. For instance, if the ISA has 128-bit
  // loads and each intrinsic consumes only 32 bits from A and B, then we want
  // to set intrinsicsK=4 to turn 4 separate 32-bit loads into one 128-bit load.
  int intrinsicLoadBits =
      std::min(sizeInBits(intrinsicA), sizeInBits(intrinsicB));
  const int intrinsicsK =
      std::max(1, *wgp.getMaxLoadInstructionBits() / intrinsicLoadBits);

  // The total amount of unrolling along the M and N dimensions is normally
  // limited only by the number of available registers, since larger M and N
  // yields higher arithmetic intensity. Here, we do not yet distinguish between
  // plain unrolling (more instructions on each thread) and
  // unrolling-to-subgroups (more threads), since expanding to more subgroups
  // correspondingly divides the available register space between this many
  // subgroups, making it cancel out of the equation here.
  //
  // We need to solve for two variables here, `totalUnrollM` and `totalUnrollN`,
  // constrained by a quadratic equation expressing that the A, B and C tiles
  // must fit in VGPR space.
  //
  // We also self-impose the constraint that `totalUnrollM` and `totalUnrollN`
  // are powers of 2 to avoid prematurely entering excessing fine-tuning of
  // unrolling factors.
  int totalUnrollM = 1;
  int totalUnrollN = 1;
  SmallVector<int64_t, 4> matmulDims = IREE::Encoding::getMatmulDims(encoding);
  // Constrain the upper bounds for `totalUnrollM` and `totalUnrollN`
  //
  // Example:
  //   MFMA_F32_16x16x4_F32 → mmaSizeM = 16, mmaSizeN = 16
  //   Matmul shape: MxNxK = 4x64x64
  //
  // In this case:
  //   - The matmul has M=4 rows and N=64 columns.
  //   - totalUnrollM <= ceilDiv(4, 16) = 1
  //   - totalUnrollN <= ceilDiv(64, 16) = 4
  //
  // If the matmul shape is not known or dynamic sizes, leaves the upper bounds
  // unset as the following loop will break anyway.
  const int64_t m =
      (matmulDims.size() == 4) && !ShapedType::isDynamic(matmulDims[1])
          ? llvm::divideCeil(matmulDims[1],
                             getMSize(intrinsicMma.getIntrinsic()))
          : INT64_MAX;
  const int64_t n =
      (matmulDims.size() == 4) && !ShapedType::isDynamic(matmulDims[2])
          ? llvm::divideCeil(matmulDims[2],
                             getNSize(intrinsicMma.getIntrinsic()))
          : INT64_MAX;
  // Enumerate x (powers of two).
  for (int x = 2; x <= m; x <<= 1) {
    // For this x, solve maximum feasible y.
    //
    // The register space taken is:
    //     A-tile: x * intrinsicsK * sizeInBits(intrinsicA)
    //     B-tile: y * intrinsicsK * sizeInBits(intrinsicB)
    //     C-tile: x * y * sizeInBits(intrinsicC)
    // So the equation to solve is:
    //       x * intrinsicsK * sizeInBits(intrinsicA)
    //     + y * intrinsicsK * sizeInBits(intrinsicB)
    //     + x * y * sizeInBits(intrinsicC)
    //    <= wgp.getVgprSpaceBits()
    int y =
        (*wgp.getVgprSpaceBits() - intrinsicsK * sizeInBits(intrinsicA) * x) /
        (intrinsicsK * sizeInBits(intrinsicB) + sizeInBits(intrinsicC) * x);
    if (y <= 0) {
      break;
    }
    if (y > n) {
      y = n;
    }
    // Round y down to nearest power of two.
    y = 1 << (int)std::floor(std::log2(y));
    // The goal is to maximize the product of the unrolling factors,
    // and minimize the difference between them (a heustic to balance the memory
    // loads for LHS and RHS).
    if ((x * y > totalUnrollM * totalUnrollN) ||
        ((x * y == totalUnrollM * totalUnrollN) &&
         std::abs(x - y) < std::abs(totalUnrollM - totalUnrollN))) {
      totalUnrollM = x;
      totalUnrollN = y;
    }
  }

  // Now we introduce unroll-to-subgroups. It doesn't change the overall tile
  // size, as it increases the number of subgroups but correspondingly decreases
  // the number of registers available to each subgroups. In other words, the
  // overall tile size determined above only needed to be concerned with the
  // overall number of registers, not with how they are split between subgroups.
  //
  // For now for simplicity we put all the unroll-to-subgroups in the one of M
  // or N dimension. TODO(#18851): revisit that.
  int subgroupsM = 1;
  int subgroupsN = *wgp.getSimdsPerWgp();
  if (m > n) {
    std::swap(subgroupsM, subgroupsN);
  }
  totalUnrollM = std::max(totalUnrollM, subgroupsM);
  totalUnrollN = std::max(totalUnrollN, subgroupsN);
  int intrinsicsM = totalUnrollM / subgroupsM;
  int intrinsicsN = totalUnrollN / subgroupsN;
  return DataTiledMMAAttr::get(ctx, intrinsicMma.getIntrinsic(), intrinsicsM,
                               subgroupsM, intrinsicsN, subgroupsN,
                               intrinsicsK);
}

static Operation *
lowerContractionOpToMultiMmaOp(OpBuilder &builder, linalg::LinalgOp linalgOp,
                               ValueRange operands,
                               GPUEncodingResolverAttr resolver) {
  IREE::GPU::TargetAttr targetAttr =
      getGPUTargetAttr(resolver.getConfiguration());
  if (!targetAttr) {
    return nullptr;
  }
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return nullptr;
  }
  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  IREE::GPU::DataTiledMMAInterfaceAttr dataTiledAttr =
      chooseDataTiledMMAAttr(resultEncoding.getElementTypesArray(), targetAttr,
                             resultEncoding, resolver);
  if (!dataTiledAttr) {
    LDBG() << "expect encodings on operand types";
    return nullptr;
  }

  SmallVector<AffineExpr> lhsExprs, rhsExprs, accExprs;
  int baseIdx = contractionDims->batch.empty() ? 0 : 1;
  if (baseIdx) {
    AffineExpr bExpr = builder.getAffineDimExpr(0);
    lhsExprs.push_back(bExpr);
    rhsExprs.push_back(bExpr);
    accExprs.push_back(bExpr);
  }
  AffineExpr mExpr = builder.getAffineDimExpr(baseIdx + 0);
  AffineExpr nExpr = builder.getAffineDimExpr(baseIdx + 1);
  AffineExpr kExpr = builder.getAffineDimExpr(baseIdx + 2);

  // The outer dims are all in row-major order after relayout.
  lhsExprs.append({mExpr, kExpr});
  rhsExprs.append({nExpr, kExpr});
  accExprs.append({mExpr, nExpr});
  int64_t numDims = baseIdx + 3;
  MLIRContext *ctx = builder.getContext();
  auto lhsMap = AffineMap::get(numDims, 0, lhsExprs, ctx);
  auto rhsMap = AffineMap::get(numDims, 0, rhsExprs, ctx);
  auto accMap = AffineMap::get(numDims, 0, accExprs, ctx);

  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  Location loc = linalgOp.getLoc();
  Operation *mmaOp = Codegen::InnerTiledOp::create(
      builder, loc, operands.take_front(inputs.size()),
      operands.take_back(outputs.size()),
      ArrayRef<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes,
      cast<IREE::GPU::DataTiledMMAAttr>(dataTiledAttr));
  return mmaOp;
}

struct GPUEncodingPackedLayoutMaterializerAttr
    : public PackedLayoutMaterializerAttrExternalModelBase<
          GPUEncodingPackedLayoutMaterializerAttr, GPUEncodingResolverAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<GPUEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto resolver = cast<GPUEncodingResolverAttr>(attr);
    DictionaryAttr config = resolver.getConfiguration();

    auto encoding = llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    IREE::GPU::TargetAttr gpuAttr = getGPUTargetAttr(config);
    if (!gpuAttr) {
      return info;
    }

    DataTiledMMAInterfaceAttr mma = chooseDataTiledMMAAttr(
        encoding.getElementTypesArray(), gpuAttr, encoding, resolver);
    if (!mma) {
      return info;
    }

    // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
    // based on its operand index in the matmul.
    TileMxNxK innerTile = mma.getTileMNK();
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, innerTile);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    auto fragment = static_cast<IREE::GPU::MMAFragment>(
        encoding.getOperandIndex().getInt());
    FailureOr<IREE::Codegen::TileSwizzle> maybeSwizzle =
        getEncodingSwizzle(encoding, mma, fragment);
    if (failed(maybeSwizzle)) {
      return info;
    }
    info.swizzle = std::move(maybeSwizzle.value());
    return info;
  }
};

struct GPUEncodingResolverMaterializerAttr
    : public EncodingLayoutMaterializerAttrExternalModelBase<
          GPUEncodingResolverMaterializerAttr, GPUEncodingResolverAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto resolverAttr = cast<GPUEncodingResolverAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    return lowerContractionOpToMultiMmaOp(b, linalgOp, convertedOperands,
                                          resolverAttr);
  }
};

struct GPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<GPUSerializableAttr,
                                                      GPUEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration = cast<GPUEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct GPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          GPULayoutResolverAttr, GPUEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    DictionaryAttr existingConfig =
        cast<GPUEncodingResolverAttr>(attr).getConfiguration();
    if (existingConfig) {
      configItems.append(existingConfig.getValue().begin(),
                         existingConfig.getValue().end());
    }
    if (IREE::GPU::TargetAttr targetAttr = getGPUTargetAttr(config)) {
      addConfigGPUTarget(ctx, targetAttr, configItems);
    }
    // Pass along the ukernel provider if one has been provided, so we can use
    // it to choose the data tiling layouts.
    if (IREE::Codegen::UKernelProviderInterface ukernelProvider =
            getUKernelProviderFromTarget(config)) {
      configItems.emplace_back(StringAttr::get(ctx, kUKernelProviderName),
                               ukernelProvider);
    }
    return GPUEncodingResolverAttr::get(ctx,
                                        DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return GPUEncodingResolverAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct GPUPadEncodingLayoutMaterializerAttr final
    : IREE::Encoding::LayoutMaterializerAttr::ExternalModel<
          GPUPadEncodingLayoutMaterializerAttr, GPUPaddingResolverAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    return clone(b, op, convertedResTypes, convertedOperands);
  }
};

struct GPUPadLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          GPUPadLayoutResolverAttr, GPUPaddingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(config);
    std::optional<IREE::GPU::L1CacheInfo> cache =
        IREE::GPU::getL1CacheInfo(gpuTarget);
    if (!cache) {
      return GPUPaddingResolverAttr::get(ctx, std::nullopt, std::nullopt);
    }
    return GPUPaddingResolverAttr::get(ctx, cache->cacheLineBytes,
                                       cache->cacheSets);
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    auto gpuPadLayoutAttr = cast<GPUPaddingResolverAttr>(attr);

    int64_t rank = type.getRank();
    auto noPaddingAttr =
        IREE::Encoding::PaddingAttr::getIdentityAttr(ctx, rank);
    if (!gpuPadLayoutAttr.getCacheLineBytes() ||
        !gpuPadLayoutAttr.getCacheSets()) {
      return noPaddingAttr;
    }

    auto paddingEncodingAttr =
        dyn_cast_or_null<IREE::Encoding::PaddingAttr>(type.getEncoding());
    if (!paddingEncodingAttr) {
      return nullptr;
    }

    // If all the padding values are already static, just return the padding
    // attribute as is.
    ArrayRef<int64_t> givenPadValues =
        paddingEncodingAttr.getPadding().asArrayRef();
    if (llvm::none_of(givenPadValues, ShapedType::isDynamic)) {
      return paddingEncodingAttr;
    }

    // Currently only support case where the
    // - innermost padding dimension is dynamic
    // - all other padding values are zero.
    if (llvm::any_of(givenPadValues.drop_back(),
                     [](int64_t val) { return val != 0; }) ||
        givenPadValues.back() != ShapedType::kDynamic) {
      return nullptr;
    }

    if (rank != givenPadValues.size()) {
      return nullptr;
    }
    // TODO: Support dynamic shape of the inner tensor size.
    ArrayRef<int64_t> tensorShape = type.getShape();
    if (tensorShape.back() == ShapedType::kDynamic) {
      return nullptr;
    }

    const int64_t elementBits = type.getElementTypeBitWidth();
    const int64_t cacheLineBytes = *gpuPadLayoutAttr.getCacheLineBytes();
    if (elementBits % 8 != 0 || elementBits > cacheLineBytes) {
      // We do not support unaligned element types.
      return noPaddingAttr;
    }

    // Attempt to maximize L1 cache bandwidth by engaging all cache sets.
    // We want to make sure that the reduction dimension is a multiple of the
    // cache line, but not a multiple of cache line * cache sets. This way the
    // next 'row' will start at a different cache set.
    const int64_t cacheSetSpanBytes =
        *gpuPadLayoutAttr.getCacheSets() * cacheLineBytes;
    const int64_t dimSizeInBytes = tensorShape.back() * (elementBits / 8);
    if (dimSizeInBytes < cacheSetSpanBytes) {
      // Very small dimension, leave as-is.
      return noPaddingAttr;
    }

    int64_t padBytes = 0;
    if (int64_t unalignedBytes = dimSizeInBytes % cacheLineBytes;
        unalignedBytes != 0) {
      // First, pad to the multiple of cache lines.
      padBytes += cacheLineBytes - unalignedBytes;
    }

    if ((dimSizeInBytes + padBytes) % cacheSetSpanBytes == 0) {
      // Pad by one cache line to engage all cache sets.
      padBytes += cacheLineBytes;
    }

    assert((dimSizeInBytes + padBytes) % cacheLineBytes == 0 &&
           "Incorrect pad amount");
    assert(padBytes < cacheSetSpanBytes && "Incorrect pad amount");
    int64_t numPadElements = (padBytes * 8) / elementBits;
    SmallVector<int64_t> padValues(rank, 0);
    padValues.back() = numPadElements;
    auto padLayout = IREE::Encoding::PaddingAttr::get(ctx, padValues);
    return padLayout;
  }
};

} // namespace

void registerGPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::GPU::IREEGPUDialect *dialect) {
    IREE::GPU::GPUEncodingResolverAttr::attachInterface<
        GPUEncodingPackedLayoutMaterializerAttr,
        GPUEncodingResolverMaterializerAttr, GPULayoutResolverAttr,
        GPUSerializableAttr>(*ctx);
    IREE::GPU::GPUPaddingResolverAttr::attachInterface<
        GPUPadEncodingLayoutMaterializerAttr, GPUPadLayoutResolverAttr>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU
