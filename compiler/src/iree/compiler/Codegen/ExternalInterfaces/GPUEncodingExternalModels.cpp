// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- GPUEncodingExternalModels.cpp --------------------------------------===//
//
// This file implements the IREE::Codegen::LayoutAttrInterface for GPU backends.
// Different from CPU backends, we do not tranpose narrow-N to narrow-M for a
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

#include <cfloat>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#define DEBUG_TYPE "iree-gpu-encoding-external-models"

namespace mlir::iree_compiler::IREE::GPU {

using Codegen::MaterializeEncodingInfo;
using Codegen::TileMxNxK;

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
    if (candidateMma && candidateMma.getKSize() > mma.getKSize()) {
      continue;
    }
    candidateMma = mma;
  }
  return candidateMma;
}

static DataTiledMMAAttr
chooseDataTiledMMAAttr(TypeRange eTypes, TargetAttr target,
                       Encoding::EncodingAttr encoding) {
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
  // The unrollK factor serves to allow loads from the A and B matrices to use
  // the target ISA's vector loads. For instance, if the ISA has 128-bit loads
  // and each intrinsic consumes only 32 bits from A and B, then we want to set
  // unrollK=4 to turn 4 separate 32-bit loads into one 128-bit load.
  int intrinsicLoadBits =
      std::min(sizeInBits(intrinsicA), sizeInBits(intrinsicB));
  const int unrollK =
      std::max(1, *wgp.getMaxLoadInstructionBits() / intrinsicLoadBits);

  // The total amount of unrolling along the M and N dimensions is normally
  // limited only by the number of available registers, since larger M and N
  // yields higher arithmetic intensity. Here, we do not yet distinguish between
  // plain unrolling (more instructions on each thread) and
  // unrolling-to-subgroups (more threads), since expanding to more subgroups
  // correspondingly divides the available register space between this many
  // subgroups, making it cancel out of the equation here.
  //
  // We need to solve for two variables here, unroll_m and unroll_n, constrained
  // by one quadratic equation expressing that the A, B and C tiles must fit in
  // VGPR space. Since we have only 1 constraint for two variables, we
  // self-impose a second constraint for now: that the unrolling shape should be
  // square, i.e. unrollM == unrollN.
  // TODO(#18850): that is suboptimal for narrow cases.
  //
  // Now we have only one variable, call it x, to solve for.

  // The register space taken is:
  //     A-tile: x * unrollK * sizeInBits(intrinsicA)
  //     B-tile: x * unrollK * sizeInBits(intrinsicB)
  //     C-tile: x^2 * sizeInBits(intrinsicC)
  // So the equation to solve is:
  //       x^2 * sizeInBits(intrinsicC)
  //     + x   * unrollK * (sizeInBits(intrinsicA) + sizeInBits(intrinsicB))
  //    == wgp.getVgprSpaceBits()
  float c2 = sizeInBits(intrinsicC);
  float c1 = unrollK * (sizeInBits(intrinsicA) + sizeInBits(intrinsicB));
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
  int unrollM = totalUnrollM / subgroupsM;
  int unrollN = totalUnrollN / subgroupsN;

  //
  // Step 3: Adjust the unrolling factors when there is a narrow dimension.
  // TODO(#18850): dealing with narrow cases as a fix-up is suboptimal.
  //
  IREE::Encoding::MatmulNarrowDim narrowDim =
      IREE::Encoding::getMatmulNarrowDim(encoding);
  if (narrowDim.isM()) {
    unrollM = std::min(unrollM, static_cast<int>(llvm::divideCeil(
                                    narrowDim.size, intrinsicMma.getMSize())));
  }
  if (narrowDim.isN()) {
    std::swap(unrollM, unrollN);
    std::swap(subgroupsM, subgroupsN);
    assert(subgroupsN == 1);
    unrollN = std::min(unrollN, static_cast<int>(llvm::divideCeil(
                                    narrowDim.size, intrinsicMma.getNSize())));
  }

  return DataTiledMMAAttr::get(ctx, intrinsicMma.getIntrinsic(), unrollM,
                               subgroupsM, unrollN, subgroupsN, unrollK);
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

struct GPUDeviceEncodingLayoutAttrInterface
    : public Codegen::LayoutAttrInterface::ExternalModel<
          GPUDeviceEncodingLayoutAttrInterface, GPUEncodingLayoutAttr> {
  MaterializeEncodingInfo getEncodingInfo(Attribute attr,
                                          RankedTensorType type) const {
    auto layoutAttr = cast<GPUEncodingLayoutAttr>(attr);
    auto encoding = llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    DataTiledMMAAttr mma = chooseDataTiledMMAAttr(
        encoding.getElementTypesArray(), layoutAttr.getTargetAttr(), encoding);
    if (!mma) {
      return info;
    }

    // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
    // based on its operand index in the matmul.
    TileMxNxK innerTile;
    std::tie(innerTile.M, innerTile.N, innerTile.K) = mma.getMNKShape();
    info = getEncodingInfoForMatmul(encoding, innerTile);
    auto fragment = static_cast<IREE::GPU::MMAFragment>(
        encoding.getOperandIndex().getInt());
    info.swizzle = getSwizzle(mma, fragment);
    return info;
  }

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<GPUEncodingLayoutAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    return lowerContractionOpToMultiMmaOp(b, linalgOp, convertedOperands,
                                          layoutAttr.getTargetAttr());
  }
};

} // namespace

void registerGPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::GPU::IREEGPUDialect *dialect) {
        IREE::GPU::GPUEncodingLayoutAttr::attachInterface<
            GPUDeviceEncodingLayoutAttrInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::GPU
