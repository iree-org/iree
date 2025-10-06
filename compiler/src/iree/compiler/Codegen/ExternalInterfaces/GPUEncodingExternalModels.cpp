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
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
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
using IREE::Codegen::TileMxNxKxKb;

namespace {

/// Chooses a ScaledMMAAttr that supports the given element types. Currently
/// just selects the first ScaledMMAAttr that is compatible with the given
/// element types.
/// TODO(#21923): This is a placeholder for now. We want a better heuristic
/// in the future.
static ScaledMMAAttr chooseScaledIntrinsicMMAAttr(TypeRange eTypes,
                                                  TargetWgpAttr wgp) {
  ScaledMMAAttr candidateMma;
  for (ScaledMMAAttr mma : wgp.getScaledMma()) {
    // Filter out intrinsics that don't match the element types of this matmul.
    if (mma.getLhsElemType() != eTypes[0] ||
        mma.getRhsElemType() != eTypes[1] ||
        mma.getAccElemType() != eTypes[4]) {
      continue;
    }
    candidateMma = mma;
    break;
  }
  return candidateMma;
}

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
  // Step 1: select a MMAIntrinsic and compute the LHS and RHS vector sizes.
  //
  auto sizeInBits = [](VectorType type) -> int64_t {
    return type.getElementTypeBitWidth() * type.getNumElements();
  };
  int64_t intrinsicSizeBitsLHS = 0;
  int64_t intrinsicSizeBitsRHS = 0;
  int64_t intrinsicSizeBitsACC = 0;
  int64_t intrinsicMSize = 0;
  int64_t intrinsicNSize = 0;
  Attribute intrinsicAttr;
  switch (encoding.getOpType().getValue()) {
  case IREE::Encoding::EncodingOpType::matmul: {
    MMAAttr intrinsicMma = chooseIntrinsicMMAAttr(eTypes, wgp);
    if (!intrinsicMma) {
      return {};
    }
    auto [intrinsicA, intrinsicB, intrinsicC] =
        intrinsicMma.getABCVectorTypes();
    intrinsicSizeBitsLHS = sizeInBits(intrinsicA);
    intrinsicSizeBitsRHS = sizeInBits(intrinsicB);
    intrinsicSizeBitsACC = sizeInBits(intrinsicC);
    intrinsicMSize = getMSize(intrinsicMma.getIntrinsic());
    intrinsicNSize = getNSize(intrinsicMma.getIntrinsic());
    intrinsicAttr = intrinsicMma;
    break;
  }
  case IREE::Encoding::EncodingOpType::scaled_matmul: {
    ScaledMMAAttr intrinsicScaledMma =
        chooseScaledIntrinsicMMAAttr(eTypes, wgp);
    if (!intrinsicScaledMma) {
      return {};
    }
    SmallVector<VectorType> vectorTypes;
    intrinsicScaledMma.getDistributedTileTypes(vectorTypes);
    // For scaled_matmul, the size of the LHS scales and RHS scales are added
    // to the total LHS and RHS sizes, because we use these sizes to select the
    // unrolling factors for M, N, and K, which affect both the input and the
    // scale operands.
    intrinsicSizeBitsLHS =
        sizeInBits(vectorTypes[0]) + sizeInBits(vectorTypes[1]);
    intrinsicSizeBitsRHS =
        sizeInBits(vectorTypes[2]) + sizeInBits(vectorTypes[3]);
    intrinsicSizeBitsACC = sizeInBits(vectorTypes[4]);
    intrinsicMSize = getMSize(intrinsicScaledMma.getIntrinsic());
    intrinsicNSize = getNSize(intrinsicScaledMma.getIntrinsic());
    intrinsicAttr = intrinsicScaledMma;
    break;
  }
  default:
    return {};
  }

  //
  // Step 2: Select the unrolling factors for the generic case where there is no
  //         narrow dimension.
  //
  // The intrinsicsK factor serves to allow loads from the A and B matrices to
  // use the target ISA's vector loads. For instance, if the ISA has 128-bit
  // loads and each intrinsic consumes only 32 bits from A and B, then we want
  // to set intrinsicsK=4 to turn 4 separate 32-bit loads into one 128-bit load.
  int intrinsicLoadBits = std::min(intrinsicSizeBitsLHS, intrinsicSizeBitsRHS);
  int intrinsicsK =
      std::max(1, *wgp.getMaxLoadInstructionBits() / intrinsicLoadBits);

  // For scaled intrinsics, there is another reason to unroll K. Scales are held
  // in a vector of multiple scales, but only a single scale is used for each
  // instruction. We want to be able to load a contiguous vector of scales into
  // registers, and use the same vector for consecutive instructions. Choose the
  // LCM of the scales vector size unrolling factor, and the load bitwidth
  // unrolling factor, so both are satisfied.
  // * Note that typically, the load bitwidth unrolling factor will be 1, so the
  // total K unrolling factor will just be the scales vector size.
  if (auto scaledMmaAttr = dyn_cast<ScaledMMAAttr>(intrinsicAttr)) {
    intrinsicsK = std::lcm(intrinsicsK, scaledMmaAttr.getScalesVectorSize());
  }

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
  float c2 = intrinsicSizeBitsACC;
  float c1 = intrinsicsK * (intrinsicSizeBitsLHS + intrinsicSizeBitsRHS);
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
    intrinsicsM = std::min(intrinsicsM, static_cast<int>(llvm::divideCeil(
                                            narrowDim.size, intrinsicMSize)));
  }
  if (narrowDim.isN()) {
    std::swap(intrinsicsM, intrinsicsN);
    std::swap(subgroupsM, subgroupsN);
    assert(subgroupsN == 1);
    intrinsicsN = std::min(intrinsicsN, static_cast<int>(llvm::divideCeil(
                                            narrowDim.size, intrinsicNSize)));
  }

  if (auto intrinsicMma = dyn_cast<MMAAttr>(intrinsicAttr)) {
    return DataTiledMMAAttr::get(ctx, intrinsicMma.getIntrinsic(), intrinsicsM,
                                 subgroupsM, intrinsicsN, subgroupsN,
                                 intrinsicsK);
  }
  auto intrinsicScaledMma = cast<ScaledMMAAttr>(intrinsicAttr);
  return DataTiledScaledMMAAttr::get(
      ctx, intrinsicScaledMma.getIntrinsic(),
      intrinsicScaledMma.getLhsElemType(), intrinsicScaledMma.getRhsElemType(),
      intrinsicScaledMma.getAccElemType(), intrinsicsM, subgroupsM, intrinsicsN,
      subgroupsN, intrinsicsK);
}

static Operation *lowerContractionOrScaledContractionOpToInnerTiledOp(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    GPUEncodingResolverAttr resolver) {
  IREE::GPU::TargetAttr targetAttr =
      getGPUTargetAttr(resolver.getConfiguration());
  if (!targetAttr) {
    return nullptr;
  }
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }
  if (!linalg::isaContractionOpInterface(linalgOp) &&
      !IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
    return nullptr;
  }

  SmallVector<Value> inputs = linalgOp.getDpsInputs();
  SmallVector<Value> outputs = linalgOp.getDpsInits();

  SmallVector<IREE::Encoding::EncodingAttr> operandEncodings;
  // Return false if the operand has no encoding.
  auto appendEncodingIfPresent = [&](Value operand) -> bool {
    auto type = cast<RankedTensorType>(operand.getType());
    auto encoding = IREE::Encoding::getEncodingAttr(type);
    if (!encoding) {
      return false;
    }
    operandEncodings.push_back(encoding);
    return true;
  };
  if (!llvm::all_of(llvm::concat<Value>(inputs, outputs),
                    appendEncodingIfPresent)) {
    return nullptr;
  }

  auto checkEncodingIndex = [&](int64_t idx, int64_t expectedIdx) -> bool {
    return operandEncodings[idx].getOperandIndex().getInt() == expectedIdx;
  };
  switch (operandEncodings[0].getOpType().getValue()) {
  case IREE::Encoding::EncodingOpType::matmul: {
    if (!checkEncodingIndex(0, IREE::Encoding::MATMUL_LHS) ||
        !checkEncodingIndex(1, IREE::Encoding::MATMUL_RHS) ||
        !checkEncodingIndex(2, IREE::Encoding::MATMUL_RESULT)) {
      return nullptr;
    }
    break;
  }
  case IREE::Encoding::EncodingOpType::scaled_matmul: {
    if (!checkEncodingIndex(0, IREE::Encoding::SCALED_MATMUL_LHS) ||
        !checkEncodingIndex(1, IREE::Encoding::SCALED_MATMUL_RHS) ||
        !checkEncodingIndex(2, IREE::Encoding::SCALED_MATMUL_LHS_SCALES) ||
        !checkEncodingIndex(3, IREE::Encoding::SCALED_MATMUL_RHS_SCALES) ||
        !checkEncodingIndex(4, IREE::Encoding::SCALED_MATMUL_RESULT)) {
      return nullptr;
    }
    break;
  }
  default:
    return nullptr;
  }

  IREE::Encoding::EncodingAttr resultEncoding = operandEncodings.back();
  IREE::GPU::DataTiledMMAInterfaceAttr dataTiledAttr =
      chooseDataTiledMMAAttr(resultEncoding.getElementTypesArray(), targetAttr,
                             resultEncoding, resolver);
  if (!dataTiledAttr) {
    LDBG() << "expect encodings on operand types";
    return nullptr;
  }

  SmallVector<AffineExpr> lhsExprs, rhsExprs, accExprs;
  Codegen::EncodingContractionLikeDimInfo cDimInfo =
      Codegen::getEncodingContractionLikeDims(resultEncoding).value();
  int numDims = 0;
  AffineExpr bExpr = builder.getAffineDimExpr(numDims);
  if (cDimInfo.batchDim.operandIdx.has_value()) {
    lhsExprs.push_back(bExpr);
    rhsExprs.push_back(bExpr);
    accExprs.push_back(bExpr);
    ++numDims;
  }
  AffineExpr mExpr = builder.getAffineDimExpr(numDims++);
  AffineExpr nExpr = builder.getAffineDimExpr(numDims++);
  AffineExpr kExpr = builder.getAffineDimExpr(numDims++);

  // The outer dims are all in row-major order after relayout.
  lhsExprs.append({mExpr, kExpr});
  rhsExprs.append({nExpr, kExpr});
  accExprs.append({mExpr, nExpr});
  SmallVector<AffineMap> indexingMaps;
  MLIRContext *ctx = builder.getContext();
  Location loc = linalgOp.getLoc();
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  switch (resultEncoding.getOpType().getValue()) {
  case IREE::Encoding::EncodingOpType::matmul: {
    indexingMaps.push_back(AffineMap::get(numDims, 0, lhsExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, rhsExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, accExprs, ctx));
    return Codegen::InnerTiledOp::create(
        builder, loc, operands.take_front(inputs.size()),
        operands.take_back(outputs.size()), indexingMaps, iteratorTypes,
        cast<IREE::GPU::DataTiledMMAAttr>(dataTiledAttr));
  }
  case IREE::Encoding::EncodingOpType::scaled_matmul: {
    SmallVector<AffineExpr> lhsScalesExprs, rhsScalesExprs;
    if (cDimInfo.batchDim.operandIdx.has_value()) {
      lhsScalesExprs.push_back(bExpr);
      rhsScalesExprs.push_back(bExpr);
    }
    AffineExpr kbExpr = builder.getAffineDimExpr(numDims++);
    lhsExprs.append({kbExpr});
    rhsExprs.append({kbExpr});
    lhsScalesExprs.append({mExpr, kExpr});
    rhsScalesExprs.append({nExpr, kExpr});
    indexingMaps.push_back(AffineMap::get(numDims, 0, lhsExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, rhsExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, lhsScalesExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, rhsScalesExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, accExprs, ctx));
    return Codegen::InnerTiledOp::create(
        builder, loc, operands.take_front(inputs.size()),
        operands.take_back(outputs.size()), indexingMaps, iteratorTypes,
        cast<IREE::GPU::DataTiledScaledMMAAttr>(dataTiledAttr));
  }
  default: {
    assert(false && "unexpected encoding op type");
    return nullptr;
  }
  }
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

    // Map the matmul TileMxNxKxKb to an actual tile shape for the tensor at
    // hand, based on its operand index in the matmul.
    TileMxNxKxKb innerTile = mma.getTileMNKKb();
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, innerTile);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    FailureOr<IREE::Codegen::TileSwizzle> maybeSwizzle =
        getEncodingSwizzle(encoding, mma, encoding.getOperandIndex().getInt());
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
    return lowerContractionOrScaledContractionOpToInnerTiledOp(
        b, linalgOp, convertedOperands, resolverAttr);
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
