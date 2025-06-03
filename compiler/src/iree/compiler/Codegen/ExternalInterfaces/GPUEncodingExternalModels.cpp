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
//   1. As linalg.matmul materializes into iree_gpu.multi_mma, which inherits
//      its semantics from the wrapped intrinsic, we can't rely on any kind of
//      LHS<->RHS symmetry.
//   2. We do not currently use ukernels, which would be one of the main areas
//      to benefit from transposeNarrowN.
//   3. Heuristics for cache-friendly dispatch tiling are internal to the GPU
//      runtime, so we don't need a simplification at that level either.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/GPUEncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
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
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <numeric>

#define DEBUG_TYPE "iree-gpu-encoding-external-models"

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

static DataTiledMMAAttr
chooseDataTiledMMAAttr(TypeRange eTypes, TargetAttr target,
                       IREE::Encoding::EncodingAttr encoding) {
  if (!target) {
    return {};
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
  // We need to solve for two variables here, intrinsics_m and intrinsics_n,
  // constrained by one quadratic equation expressing that the A, B and C tiles
  // must fit in VGPR space. Since we have only 1 constraint for two variables,
  // we self-impose a second constraint for now: that the unrolling shape should
  // be square, i.e. intrinsicsM == intrinsicsN.
  // TODO(#18850): that is suboptimal for narrow cases.
  //
  // Now we have only one variable, call it x, to solve for.

  // The register space taken is:
  //     A-tile: x * intrinsicsK * sizeInBits(intrinsicA)
  //     B-tile: x * intrinsicsK * sizeInBits(intrinsicB)
  //     C-tile: x^2 * sizeInBits(intrinsicC)
  // So the equation to solve is:
  //       x^2 * sizeInBits(intrinsicC)
  //     + x   * intrinsicsK * (sizeInBits(intrinsicA) + sizeInBits(intrinsicB))
  //    == wgp.getVgprSpaceBits()
  float c2 = sizeInBits(intrinsicC);
  float c1 = intrinsicsK * (sizeInBits(intrinsicA) + sizeInBits(intrinsicB));
  float c0 = -*wgp.getVgprSpaceBits(); // negative by construction.
  // Now the equation to solve is: c2 * x^2 + c1 * x + c0 == 0.
  float discriminant = c1 * c1 - 4 * c0 * c2; // positive, because c0 < 0.
  // x = unique positive solution.
  float x = (-c1 + std::sqrt(discriminant)) / (2 * c2);

#ifndef NDEBUG
  // Self-check quadratic solver. 10 epsilon is just a crude upper bound;
  // In practice, cancellation results in check == 0 in current cases.
  float check = c2 * x * x + c1 * x + c0;
  assert(std::abs(check) < 10 * FLT_EPSILON * std::abs(c0));
#endif

  // Now, looking geometrically at our unrolling space along the M and N
  // dimensions, we solve the following problem in the (M,N)-plane: approximate
  // a square of side length `x`, by a rectangle of side lengths `totalUnrollM`
  // and `totalUnrollN`, under the constraints:
  // 1. totalUnrollM * totalUnrollN <= x * x
  //    * Reason: by construction of x, any larger area would exceed the
  //      wgp.getVgprSpaceBits() budget.
  // 2. totalUnrollM and totalUnrollN are powers of 2.
  //    * Reason: that is a self-imposed constraint for now to avoid prematurely
  //      entering excessing fine-tuning of unrolling factors. Also, since below
  //      we will put all the unroll-to-subgroups in the N dimension, that
  //      requires totalUnrollN to be a multiple of wgp.getSimdsPerWgp(),
  //      which is typically a power of 2, specifically 4.
  //      TODO(#18851): we will not always put all the unroll-to-subgroups on N.
  // 3. totalUnrollN >= totalUnrollM.
  //    * Reason: Just like the previous constraint, that is also motivated by
  //      the code below currently putting all the unroll-to-subgroups in the N
  //      dimension, which requires a sufficiently large totalUnrollN.
  //      TODO(#18851): we will not always put all the unroll-to-subgroups on N.
  //
  // Set totalUnrollN = round x to nearest power of two, break ties away from 0
  // per specification of std::round.
  int totalUnrollN = std::exp2(std::round(std::log2(x)));
  // Based on above constraint 1:
  float unroundedMaxTotalUnrollM = x * x / totalUnrollN;
  int totalUnrollM = std::exp2(std::floor(std::log2(unroundedMaxTotalUnrollM)));

  // Now we introduce unroll-to-subgroups. It doesn't change the overall tile
  // size, as it increases the number of subgroups but correspondingly decreases
  // the number of registers available to each subgroups. In other words, the
  // overall tile size determined above only needed to be concerned with the
  // overall number of registers, not with how they are split between subgroups.
  //
  // For now for simplicity we put all the unroll-to-subgroups in the N
  // dimension. TODO(#18851): revisit that.
  //
  // That does simplify the below adjustments for narrow M/N, as we don't need
  // to think about unroll-to-subgroups when making the narrowing adjustment.
  int subgroupsM = 1;
  int subgroupsN = *wgp.getSimdsPerWgp();
  int intrinsicsM = totalUnrollM / subgroupsM;
  int intrinsicsN = totalUnrollN / subgroupsN;

  //
  // Step 3: Adjust the unrolling factors when there is a narrow dimension.
  // TODO(#18850): dealing with narrow cases as a fix-up is suboptimal.
  //
  IREE::Encoding::MatmulNarrowDim narrowDim =
      IREE::Encoding::getPo2MatmulNarrowDim(encoding);
  if (narrowDim.isM()) {
    intrinsicsM =
        std::min(intrinsicsM,
                 static_cast<int>(llvm::divideCeil(
                     narrowDim.size, getMSize(intrinsicMma.getIntrinsic()))));
  }
  if (narrowDim.isN()) {
    std::swap(intrinsicsM, intrinsicsN);
    std::swap(subgroupsM, subgroupsN);
    assert(subgroupsN == 1);
    intrinsicsN =
        std::min(intrinsicsN,
                 static_cast<int>(llvm::divideCeil(
                     narrowDim.size, getNSize(intrinsicMma.getIntrinsic()))));
  }

  return DataTiledMMAAttr::get(ctx, intrinsicMma.getIntrinsic(), intrinsicsM,
                               subgroupsM, intrinsicsN, subgroupsN,
                               intrinsicsK);
}

static Operation *lowerContractionOpToMultiMmaOp(OpBuilder &builder,
                                                 linalg::LinalgOp linalgOp,
                                                 ValueRange operands,
                                                 TargetAttr targetAttr) {
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

  IREE::GPU::DataTiledMMAAttr mma = chooseDataTiledMMAAttr(
      resultEncoding.getElementTypesArray(), targetAttr, resultEncoding);
  if (!mma) {
    LLVM_DEBUG(llvm::dbgs() << "expect encodings on operand types\n");
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "Target MMA: " << mma << "\n");

  MLIRContext *ctx = builder.getContext();
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
  auto lhsMap = AffineMap::get(numDims, 0, lhsExprs, ctx);
  auto rhsMap = AffineMap::get(numDims, 0, rhsExprs, ctx);
  auto accMap = AffineMap::get(numDims, 0, accExprs, ctx);

  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  Location loc = linalgOp.getLoc();
  Operation *mmaOp = builder.create<MultiMmaOp>(
      loc, operands[0], operands[1], operands[2],
      ArrayRef<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes, mma);
  return mmaOp;
}

struct GPUEncodingPackedLayoutMaterializerAttr
    : public PackedLayoutMaterializerAttrExternalModelBase<
          GPUEncodingPackedLayoutMaterializerAttr, GPUEncodingLayoutAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<GPUEncodingLayoutAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<GPUEncodingLayoutAttr>(attr);
    DictionaryAttr config = layoutAttr.getConfiguration();

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
    DataTiledMMAAttr mma = chooseDataTiledMMAAttr(
        encoding.getElementTypesArray(), gpuAttr, encoding);
    if (!mma) {
      return info;
    }

    // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
    // based on its operand index in the matmul.
    TileMxNxK innerTile;
    std::tie(innerTile.M, innerTile.N, innerTile.K) = mma.getMNKShape();
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

struct GPUEncodingLayoutMaterializerAttr
    : public EncodingLayoutMaterializerAttrExternalModelBase<
          GPUEncodingLayoutMaterializerAttr, GPUEncodingLayoutAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<GPUEncodingLayoutAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    DictionaryAttr config = layoutAttr.getConfiguration();
    IREE::GPU::TargetAttr gpuAttr = getGPUTargetAttr(config);
    if (!gpuAttr) {
      return nullptr;
    }
    return lowerContractionOpToMultiMmaOp(b, linalgOp, convertedOperands,
                                          gpuAttr);
  }
};

struct GPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<GPUSerializableAttr,
                                                      GPUEncodingLayoutAttr> {

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct GPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<GPULayoutResolverAttr,
                                                        GPUEncodingLayoutAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    DictionaryAttr existingConfig =
        cast<GPUEncodingLayoutAttr>(attr).getConfiguration();
    if (existingConfig) {
      configItems.append(existingConfig.getValue().begin(),
                         existingConfig.getValue().end());
    }
    storeNamedAttrIfPresent(configItems, config, kGPUTargetAttrName);
    return GPUEncodingLayoutAttr::get(ctx,
                                      DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return GPUEncodingLayoutAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct GPUPadEncodingLayoutMaterializerAttr final
    : IREE::Encoding::LayoutMaterializerAttr::ExternalModel<
          GPUPadEncodingLayoutMaterializerAttr, GPUPadLayoutAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    return clone(b, op, convertedResTypes, convertedOperands);
  }
};

struct GPUPadLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          GPUPadLayoutResolverAttr, GPUPadLayoutAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(config);
    std::optional<IREE::GPU::L1CacheInfo> cache =
        IREE::GPU::getL1CacheInfo(gpuTarget);
    if (!cache) {
      return GPUPadLayoutAttr::get(ctx, std::nullopt, std::nullopt);
    }
    return GPUPadLayoutAttr::get(ctx, cache->cacheLineBytes, cache->cacheSets);
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    auto gpuPadLayoutAttr = cast<GPUPadLayoutAttr>(attr);

    int64_t rank = type.getRank();
    auto noPaddingAttr =
        IREE::Encoding::PadEncodingLayoutAttr::getIdentityAttr(ctx, rank);
    if (!gpuPadLayoutAttr.getCacheLineBytes() ||
        !gpuPadLayoutAttr.getCacheSets()) {
      return noPaddingAttr;
    }

    auto paddingEncodingAttr =
        dyn_cast_or_null<IREE::Encoding::PadEncodingLayoutAttr>(
            type.getEncoding());
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
    auto padLayout = IREE::Encoding::PadEncodingLayoutAttr::get(ctx, padValues);
    return padLayout;
  }
};

} // namespace

void registerGPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::GPU::IREEGPUDialect *dialect) {
    IREE::GPU::GPUEncodingLayoutAttr::attachInterface<
        GPUEncodingPackedLayoutMaterializerAttr,
        GPUEncodingLayoutMaterializerAttr, GPULayoutResolverAttr,
        GPUSerializableAttr>(*ctx);
    IREE::GPU::GPUPadLayoutAttr::attachInterface<
        GPUPadEncodingLayoutMaterializerAttr, GPUPadLayoutResolverAttr>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU
