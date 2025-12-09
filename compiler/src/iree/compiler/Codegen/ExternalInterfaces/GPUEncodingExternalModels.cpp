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
// - VerifiableTensorEncoding
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
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TensorEncoding.h"

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
    intrinsicSizeBitsLHS = sizeInBits(vectorTypes[kScaledMMAOperandLhs]) +
                           sizeInBits(vectorTypes[kScaledMMAOperandLhsScale]);
    intrinsicSizeBitsRHS = sizeInBits(vectorTypes[kScaledMMAOperandRhs]) +
                           sizeInBits(vectorTypes[kScaledMMAOperandRhsScale]);
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
  // Step 2: Select the total unrolling factors along the M, N, and K
  // dimensions.
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
  // We need to find the optimal pair (totalUnrollM, totalUnrollN) by
  // enumerating feasible (tm, tn) candidates. For each candidate, the
  // following two constraints are enforced:
  // 1. The A, B and C tiles must fit in VGPR space.
  //     A-tile + B-tile + C-tile <= wgp.getVgprSpaceBits()
  // 2. The A, B tiles must fit in shared memory.
  //     A-tile + B-tile <= wgp.getMaxWorkgroupMemoryBytes() * 8
  // A-tile: tm * intrinsicsK * intrinsicSizeBitsLHS
  // B-tile: tn * intrinsicsK * intrinsicSizeBitsRHS
  // C-tile: tm * tn * intrinsicSizeBitsACC
  //
  // The optimization goal is to maximize arithmetic intensity (tm * tn) / (tm +
  // tn).
  //
  // We also self-impose the constraint that tm and tn are powers of 2 to
  // avoid prematurely entering excessing fine-tuning of unrolling factors.
  int64_t totalUnrollM = 1;
  int64_t totalUnrollN = 1;
  auto computeArithmeticIntensity = [&](int64_t tm, int64_t tn) -> double {
    return double(tm * tn) / double(tm + tn);
  };
  double bestArithmeticIntensity =
      computeArithmeticIntensity(totalUnrollM, totalUnrollN);
  // Upper bounds of tm and tn are decided by the matrix and intrinsic sizes.
  int64_t maxTotalUnrollM = INT64_MAX;
  int64_t maxTotalUnrollN = INT64_MAX;
  // Upper bound of tm * tn are decided by the workgroup count of the chip and
  // the intrinsic sizes.
  int64_t maxTotalUnrollMN = INT64_MAX;
  FailureOr<IREE::Encoding::BxMxNxKxKb> matmulSizes =
      getEncodingContractionLikeSizes(encoding);
  if (succeeded(matmulSizes)) {
    if (!ShapedType::isDynamic(matmulSizes->M)) {
      // Cap maxTotalUnrollM to avoid excessive padding.
      maxTotalUnrollM = llvm::divideCeil(matmulSizes->M, intrinsicMSize);
    }
    if (!ShapedType::isDynamic(matmulSizes->N)) {
      // Cap maxTotalUnrollN to avoid excessive padding.
      maxTotalUnrollN = llvm::divideCeil(matmulSizes->N, intrinsicNSize);
    }
    if (!ShapedType::isDynamic(matmulSizes->M) &&
        !ShapedType::isDynamic(matmulSizes->N)) {
      // Cap maxTotalUnrollMN to avoid underutilizing the workgroups available.
      IREE::GPU::TargetChipAttr chip = target.getChip();
      int64_t numWGPs = chip ? chip.getWgpCount() : 512;
      maxTotalUnrollMN =
          llvm::divideCeil(matmulSizes->M * matmulSizes->N,
                           numWGPs * intrinsicMSize * intrinsicNSize);
    }
  }
  // Iterate over possible tm.
  for (int64_t tm = 1; tm <= maxTotalUnrollM; tm <<= 1) {
    // Compute the maximum feasible tn for this tm.
    int64_t maxFeasibleTnVgpr =
        (*wgp.getVgprSpaceBits() - tm * intrinsicsK * intrinsicSizeBitsLHS) /
        (intrinsicsK * intrinsicSizeBitsRHS + tm * intrinsicSizeBitsACC);
    int64_t maxFeasibleTnSharedMem = (wgp.getMaxWorkgroupMemoryBytes() * 8 -
                                      tm * intrinsicsK * intrinsicSizeBitsLHS) /
                                     (intrinsicsK * intrinsicSizeBitsRHS);
    int64_t tn = std::min(maxFeasibleTnVgpr, maxFeasibleTnSharedMem);
    // Clamp tn to maxTotalUnrollN.
    tn = std::min(tn, maxTotalUnrollN);
    // Clamp tn to maxTotalUnrollMN / tm.
    tn = std::min(tn, maxTotalUnrollMN / tm);
    // No feasible tn for this tm. Stop the enumeration.
    if (tn <= 0) {
      break;
    }
    // Round tn down to nearest power of two.
    tn = 1 << (int64_t)std::floor(std::log2(tn));
    // Maximize arithmetic intensity (tm * tn) / (tm + tn).
    double currentArithmeticIntensity = computeArithmeticIntensity(tm, tn);
    if (currentArithmeticIntensity > bestArithmeticIntensity) {
      totalUnrollM = tm;
      totalUnrollN = tn;
      bestArithmeticIntensity = currentArithmeticIntensity;
    }
  }

  //
  // Step 3: Split `totalUnrollM` and `totalUnrollN` into plain unrolling (more
  // instructions on each thread) and unrolling-to-subgroups (more threads).
  //
  // Unrolling-to-subgroups doesn't change the overall tile
  // size, as it increases the number of subgroups but correspondingly decreases
  // the number of registers available to each subgroups. In other words, the
  // overall tile size determined above only needed to be concerned with the
  // overall number of registers, not with how they are split between subgroups.
  //
  // The goal is still to maximize arithmetic intensity, but now we need to
  // optimize `intrinsicsM(N)` instead of `totalUnrollM(N)`.
  int64_t subgroupsM = 1;
  int64_t subgroupsN = 1;
  int64_t intrinsicsM = 1;
  int64_t intrinsicsN = 1;
  bestArithmeticIntensity =
      computeArithmeticIntensity(intrinsicsM, intrinsicsN);
  int64_t simdsPerWgp = *wgp.getSimdsPerWgp();
  // Enumerate possible unrolling-to-subgroups on M dimension.
  for (int64_t sm = 1; sm <= std::min(simdsPerWgp, totalUnrollM); sm <<= 1) {
    // Calculate the unrolling-to-subgroups on N dimension, given the current
    // sm.
    int64_t sn = std::min(simdsPerWgp / sm, totalUnrollN);
    // Round sn down to nearest power of two.
    sn = 1 << (int64_t)std::floor(std::log2(sn));
    // Calculate the plain (intrinsic) unrolling factors on M and N dimensions.
    int64_t im = totalUnrollM / sm;
    int64_t in = totalUnrollN / sn;
    // Maximize arithmetic intensity (im * in) / (im + in).
    double currentArithmeticIntensity = computeArithmeticIntensity(im, in);
    if (currentArithmeticIntensity > bestArithmeticIntensity) {
      subgroupsM = sm;
      subgroupsN = sn;
      intrinsicsM = im;
      intrinsicsN = in;
      bestArithmeticIntensity = currentArithmeticIntensity;
    }
  }

  // We currently never generate subgroupsK != 1, as subgroupsK requires
  // specific partial-accumulator-reduction in the kernel, currently only done
  // in some microkernels that would provide their own DataTiledMMAAttr.
  int subgroupsK = 1;

  // Returns the final choice of attributes.
  if (auto intrinsicMma = dyn_cast<MMAAttr>(intrinsicAttr)) {
    // For non-scaled matmuls, the purpose of unrolling on K is to allow LHS/RHS
    // loads to match the preferred load instruction size. This is achieved by
    // enabling interleaving for those operands.
    auto mmaInterleaveK =
        DenseI64ArrayAttr::get(ctx, {kMMAOperandLhs, kMMAOperandRhs});
    return DataTiledMMAAttr::get(
        ctx, intrinsicMma.getIntrinsic(), intrinsicsM, subgroupsM, intrinsicsN,
        subgroupsN, intrinsicsK, subgroupsK,
        /*operands_interleaving_intrinsics_m=*/{},
        /*operands_interleaving_intrinsics_n=*/{},
        /*operands_interleaving_intrinsics_k=*/mmaInterleaveK);
  }
  // For scaled matmuls, interleaving happens because we want to load all
  // the unrolled scales with each vector load, so we need to interleave at
  // the very last dimension for the scales. For the LHS/RHS, we load in blocks,
  // so we don't need to interleave.
  auto scaledMmaInterleaveK = DenseI64ArrayAttr::get(
      ctx, {kScaledMMAOperandLhsScale, kScaledMMAOperandRhsScale});
  auto intrinsicScaledMma = cast<ScaledMMAAttr>(intrinsicAttr);
  return DataTiledScaledMMAAttr::get(
      ctx, intrinsicScaledMma.getIntrinsic(),
      intrinsicScaledMma.getLhsElemType(), intrinsicScaledMma.getRhsElemType(),
      intrinsicScaledMma.getAccElemType(), intrinsicsM, subgroupsM, intrinsicsN,
      subgroupsN, intrinsicsK, subgroupsK,
      /*operands_interleaving_intrinsics_m=*/{},
      /*operands_interleaving_intrinsics_n=*/{},
      /*operands_interleaving_intrinsics_k=*/scaledMmaInterleaveK);
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
  auto semantics = InnerTiledSemanticsAttr::get(
      builder.getContext(), /*distributed=*/false, /*opaque=*/false);
  switch (resultEncoding.getOpType().getValue()) {
  case IREE::Encoding::EncodingOpType::matmul: {
    indexingMaps.push_back(AffineMap::get(numDims, 0, lhsExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, rhsExprs, ctx));
    indexingMaps.push_back(AffineMap::get(numDims, 0, accExprs, ctx));
    return Codegen::InnerTiledOp::create(
        builder, loc, operands.take_front(inputs.size()),
        operands.take_back(outputs.size()), indexingMaps, iteratorTypes,
        cast<IREE::GPU::DataTiledMMAAttr>(dataTiledAttr), semantics);
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
        cast<IREE::GPU::DataTiledScaledMMAAttr>(dataTiledAttr), semantics);
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

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

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
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    if (linalg::isaContractionOpInterface(linalgOp) ||
        IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
      return lowerContractionOrScaledContractionOpToInnerTiledOp(
          b, linalgOp, convertedOperands, resolverAttr);
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
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

struct GPUEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<GPUEncodingResolverVerifier,
                                                    GPUEncodingResolverAttr> {
  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

// Returns the pad encoding layout, or nullptr if this is not the only layout or
// if there's no encoding at all.
static IREE::Encoding::PaddingAttr
getPadLayout(IREE::Encoding::LayoutResolverAttr layoutAttr,
             RankedTensorType type) {
  if (!type.getEncoding()) {
    return nullptr;
  }
  auto encoding =
      dyn_cast_if_present<IREE::Encoding::LayoutAttr>(type.getEncoding());
  if (encoding) {
    ArrayAttr layouts = encoding.getLayouts();
    if (layouts.size() != 1) {
      return nullptr;
    }
    return dyn_cast<IREE::Encoding::PaddingAttr>(*layouts.begin());
  }
  Attribute resolvedEncoding = layoutAttr.getLayout(type);
  LDBG() << "Unresolved type: " << type;
  LDBG() << "layoutAttr: " << layoutAttr;
  LDBG() << "Resolved into: " << resolvedEncoding;
  return dyn_cast<IREE::Encoding::PaddingAttr>(resolvedEncoding);
}

// Returns a padded tensor type (without encoding) for tensor types with the pad
// encoding layout, or the same type for all other tensors.
static RankedTensorType
getPaddedType(IREE::Encoding::LayoutResolverAttr layoutAttr,
              RankedTensorType type) {
  IREE::Encoding::PaddingAttr layout = getPadLayout(layoutAttr, type);
  if (!layout) {
    return nullptr;
  }
  if (layout.isIdentityLayout()) {
    return type.dropEncoding();
  }

  ArrayRef<int64_t> padding = layout.getPadding().asArrayRef();
  auto newShape = llvm::to_vector_of<int64_t>(type.getShape());
  for (auto [newDim, padValue] : llvm::zip_equal(newShape, padding)) {
    assert((padValue == 0 || ShapedType::isStatic(newDim)) &&
           "Padding dynamic dims not supported");
    newDim += padValue;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

struct GPUPadEncodingLayoutMaterializerAttr final
    : IREE::Encoding::LayoutMaterializerAttr::ExternalModel<
          GPUPadEncodingLayoutMaterializerAttr, GPUPaddingResolverAttr> {

  Type convertType(Attribute attr, Type type) const {
    auto layoutAttr = cast<IREE::Encoding::LayoutResolverAttr>(attr);
    return TypeSwitch<Type, Type>(type)
        .Case([](RankedTensorType type) {
          // By the definition, the final converted type is the same tensor type
          // without encodings.
          return type.dropEncoding();
        })
        .Case([&](IREE::TensorExt::DispatchTensorType dispatchTensorType) {
          auto type =
              dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
          if (!type || !type.getEncoding()) {
            return dispatchTensorType;
          }
          // The incoming bindings have the padded type, if `padding` is
          // present.
          if (getPadLayout(layoutAttr, type)) {
            type = getPaddedType(layoutAttr, type);
          }
          return IREE::TensorExt::DispatchTensorType::get(
              dispatchTensorType.getAccess(), type);
        })
        .Default([](Type type) { return type; });
  }

  LogicalResult getOffsetsSizesStrides(
      Attribute attr, OpBuilder &builder, Location loc,
      IREE::TensorExt::DispatchTensorType type, ValueRange dynamicDims,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      ArrayRef<OpFoldResult> strides, SmallVectorImpl<OpFoldResult> &newOffsets,
      SmallVectorImpl<OpFoldResult> &newSizes,
      SmallVectorImpl<OpFoldResult> &newStrides) const {
    auto boundType = dyn_cast<RankedTensorType>(type.getBoundType());
    if (!boundType || !boundType.getEncoding()) {
      return failure();
    }
    newSizes.assign(sizes.begin(), sizes.end());
    newOffsets.assign(offsets.begin(), offsets.end());
    newStrides.assign(strides.begin(), strides.end());
    return success();
  }

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
        dyn_cast_if_present<IREE::Encoding::PaddingAttr>(type.getEncoding());
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
        GPUSerializableAttr, GPUEncodingResolverVerifier>(*ctx);
    IREE::GPU::GPUPaddingResolverAttr::attachInterface<
        GPUPadEncodingLayoutMaterializerAttr, GPUPadLayoutResolverAttr>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU
