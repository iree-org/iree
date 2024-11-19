// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-llvmgpu-configure-vector-layouts"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONFIGURETENSORLAYOUTSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

using IREE::VectorExt::NestedLayoutAttr;
using IREE::VectorExt::ToLayoutOp;
using IREE::VectorExt::VectorLayoutInterface;

namespace {

static SmallVector<bool> getPromotedOperands(Operation *op) {
  SmallVector<bool> promotedOperands(op->getNumOperands(), false);

  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!config) {
    return promotedOperands;
  }

  std::optional<SmallVector<int64_t>> promoteConfig =
      config.getPromotedOperandList();
  if (!promoteConfig) {
    return promotedOperands;
  }

  for (int64_t operand : promoteConfig.value()) {
    promotedOperands[operand] = true;
  }

  return promotedOperands;
}

static IREE::GPU::MmaInterfaceAttr getIntrinsic(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  IREE::GPU::MmaInterfaceAttr mmaIntrinsic = config.getMmaKind();
  assert(mmaIntrinsic && "Cannot find intrinsic in lowering config.");
  return mmaIntrinsic;
}

static int64_t getSubgroupMCount(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  return *config.getSubgroupMCount();
}

static int64_t getSubgroupNCount(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  return *config.getSubgroupNCount();
}

/// Gets a unit vector of the given rank, but fills in the given dimensions
/// from the 2 element array |counts|. |dim0| is the position in the returned
/// vector to put the first element of |counts|, and |dim1| is the position to
/// put the second element. For example,
///
/// rank = 3, counts = [5, 7], dim0 = 2, dim1 = 1
/// returns [1, 5, 7]
static SmallVector<int64_t> getUnitOfRankWithDims(int64_t rank,
                                                  ArrayRef<int64_t> counts,
                                                  int64_t dim0, int64_t dim1) {
  assert(counts.size() == 2 &&
         "Unexpected non-rank 2 single subgroup dimension counts");
  SmallVector<int64_t> res(rank, 1);
  res[dim0] = counts[0];
  res[dim1] = counts[1];
  return res;
}

/// Constructs the nested layout given the layout for a single subgroup and the
/// subgroup/batch counts and orders, as well as the dimensions along which to
/// distribute the intrinsic's layout.
///
/// |outerDim| and |innerDim| refer to which dimensions are the outermost and
/// innermost for a canonical MK_KN_MN matrix multiply, for a particular
/// fragment. For example, for the B matrix of an MK_NK_MN matrix multiply,
/// we would have:
///   outerDim = 1 for the K dim
///   innerDim = 0 for the N dim
///
/// For something like MK_NKN_MN with multiple N dims, it would typically be:
///   outerDim = 1 for K
///   innerDim = 2 for the second N dim
///
/// Importantly these two dimensions always refer to the actual dimension
/// positions in the undistributed vector. For each fragment, this means:
///   A: [outerDim, innerDim] = [innerMostMDim, innerMostKDim]
///   B: [outerDim, innerDim] = [innerMostKDim, innerMostNDim]
///   C: [outerDim, innerDim] = [innerMostMDim, innerMostNDim]
///
/// And here inner most is referential to the iteration order, not the order
/// they appear per fragment (because there is no relationship between the
/// dimension order of M in A and in C, for example).
static NestedLayoutAttr createNestedLayout(
    MLIRContext *context, int64_t rank, int64_t outerDim, int64_t innerDim,
    ArrayRef<int64_t> subgroupSizes, ArrayRef<int64_t> subgroupStrides,
    ArrayRef<int64_t> batchCount, IREE::GPU::MMASingleSubgroupLayout counts) {

  LLVM_DEBUG({
    llvm::dbgs() << "Creating Nested Layout for::";
    llvm::dbgs() << "\n    outerDim = " << outerDim;
    llvm::dbgs() << "\n    innerDim = " << innerDim;
    llvm::dbgs() << "\n    subgroupSizes: ";
    llvm::interleaveComma(subgroupSizes, llvm::dbgs());
    llvm::dbgs() << "\n    subgroupStrides: ";
    llvm::interleaveComma(subgroupStrides, llvm::dbgs());
    llvm::dbgs() << "\n    batchCount: ";
    llvm::interleaveComma(batchCount, llvm::dbgs());
    llvm::dbgs() << "\n    counts.outer: ";
    llvm::interleaveComma(counts.outer, llvm::dbgs());
    llvm::dbgs() << "\n    counts.thread: ";
    llvm::interleaveComma(counts.thread, llvm::dbgs());
    llvm::dbgs() << "\n    counts.element: ";
    llvm::interleaveComma(counts.element, llvm::dbgs());
    llvm::dbgs() << "\n    counts.tstrides: ";
    llvm::interleaveComma(counts.tstrides, llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  SmallVector<int64_t> outerCount =
      getUnitOfRankWithDims(rank, counts.outer, outerDim, innerDim);
  SmallVector<int64_t> threadCount =
      getUnitOfRankWithDims(rank, counts.thread, outerDim, innerDim);
  SmallVector<int64_t> threadStrides =
      getUnitOfRankWithDims(rank, counts.tstrides, outerDim, innerDim);
  SmallVector<int64_t> elementCount =
      getUnitOfRankWithDims(rank, counts.element, outerDim, innerDim);

  auto layoutAttr = NestedLayoutAttr::get(context, subgroupSizes, batchCount,
                                          outerCount, threadCount, elementCount,
                                          subgroupStrides, threadStrides);
  return layoutAttr;
}

static FailureOr<std::tuple<IREE::VectorExt::VectorLayoutInterface,
                            IREE::VectorExt::VectorLayoutInterface,
                            IREE::VectorExt::VectorLayoutInterface>>
getContractionLayout(IREE::GPU::MMAScheduleAttr schedule,
                     VectorContractOpInfo &opInfo,
                     linalg::LinalgOp contractOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Getting mma layouts for:\n" << contractOp << "\n";
    llvm::dbgs() << "For schedule: " << schedule << "\n";
  });

  int64_t rank = contractOp.getIteratorTypesArray().size();
  auto mmaAttr =
      llvm::cast<IREE::GPU::MmaInterfaceAttr>(schedule.getIntrinsic());
  MLIRContext *context = schedule.getContext();

  SmallVector<int64_t> bounds = contractOp.getStaticLoopRanges();
  if (llvm::any_of(bounds,
                   [](int64_t x) { return x == ShapedType::kDynamic; })) {
    return failure();
  }

  if (!llvm::all_of(opInfo.getBatchDims(),
                    [&bounds](int64_t dim) { return bounds[dim] == 1; })) {
    LLVM_DEBUG({ llvm::dbgs() << "non-unit batch dimension\n"; });
    return failure();
  }

  // Get the concrete nested layout for each matrix. Note that the struct
  // MMASingleSubgroupLayout contains the partial layout for the
  // canonical (M, K) x (K, N) -> (M, N) matmul form; while the specific
  // contract op we are looking at right now may not be exactly in that form.
  // So here we need to permute/transpose the canonical layout to match with
  // the concrete contract op.

  // Note that no matter how we permute/transpose the input contraction
  // problem, the way we view the hardware warps remain the same--that is,
  // from the hardware's perspective, a single warp has the same warp ID no
  // matter what part of the contraction it works on. Similarly here, we are
  // delinearizing the linearized GPU hardware lane ID into a n-D concatenated
  // logical warp+thread using the subgroup/thread basis, so the subgroup
  // basis should remain the same for all A/B/C matrix.

  auto [intrinsicM, intrinsicN, intrinsicK] = mmaAttr.getMNKShape();

  SmallVector<int64_t, 2> subgroupMBasis;
  SmallVector<int64_t, 2> batchMSizes;
  int64_t currMCount = schedule.getSubgroupMCount();

  auto divideGreedily = [](int64_t availableSubgroups, int64_t dimSize,
                           int64_t minDimSize) -> std::pair<int64_t, int64_t> {
    int64_t dividableDim = dimSize / minDimSize;
    int64_t subgroupsUsed = std::gcd(availableSubgroups, dividableDim);
    dividableDim /= subgroupsUsed;
    int64_t batchesUsed = dividableDim;
    return {subgroupsUsed, batchesUsed};
  };

  // Greedily break up the M subgroup and batch counts along the "M" iteration
  // bounds. We distribute as many residual subgroups as possible per M dim,
  // and then divide the remaining along batch dims. The inner most M dim is
  // always the one used for the intrinsic, meaning for a valid schedule, the
  // computed batch counts and subgroup basis will satisfy totalMSize /
  // intrinsicM = product(batchMSizes) * product(subgroupMBasis)
  for (auto dim : opInfo.getMDims()) {
    // Get the number of subgroups and batches used for this dimension based
    // on the intrinsic size and the bound size.
    int64_t subgroupsUsed, batchesUsed;
    if (dim == opInfo.getMDims().back()) {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currMCount, bounds[dim], intrinsicM);
    } else {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currMCount, bounds[dim], 1);
    }
    subgroupMBasis.push_back(subgroupsUsed);
    batchMSizes.push_back(batchesUsed);
    // Update available subgroup count.
    currMCount /= subgroupsUsed;
  }

  SmallVector<int64_t, 2> subgroupNBasis;
  SmallVector<int64_t, 2> batchNSizes;
  int64_t currNCount = schedule.getSubgroupNCount();

  // Do the same for N dims.
  for (auto dim : opInfo.getNDims()) {
    // Get the number of subgroups and batches used for this dimension based
    // on the intrinsic size and the bound size.
    int64_t subgroupsUsed, batchesUsed;
    if (dim == opInfo.getNDims().back()) {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currNCount, bounds[dim], intrinsicN);
    } else {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currNCount, bounds[dim], 1);
    }
    subgroupNBasis.push_back(subgroupsUsed);
    batchNSizes.push_back(batchesUsed);
    // Update available subgroup count.
    currNCount /= subgroupsUsed;
  }

  SmallVector<int64_t> subgroupMStrides(subgroupMBasis.size());
  SmallVector<int64_t> subgroupNStrides(subgroupNBasis.size());

  auto mDimVec = opInfo.getMDims();
  llvm::SmallDenseSet<int64_t> mDims(mDimVec.begin(), mDimVec.end());
  auto nDimVec = opInfo.getNDims();
  llvm::SmallDenseSet<int64_t> nDims(nDimVec.begin(), nDimVec.end());
  // Because we currently require all batch dimensions to be unit, the
  // subgroup basis can be constructed from the M and N bases. To keep things
  // simple, the current heuristic is to distribute the loop dimensions from
  // outer to inner.
  int64_t currStride = 1;
  int64_t currM = subgroupMStrides.size() - 1;
  int64_t currN = subgroupNStrides.size() - 1;
  for (int64_t dim : llvm::reverse(llvm::seq<int64_t>(rank))) {
    if (mDims.contains(dim)) {
      subgroupMStrides[currM] = currStride;
      currStride *= subgroupMBasis[currM];
      currM--;
      continue;
    }

    if (nDims.contains(dim)) {
      subgroupNStrides[currN] = currStride;
      currStride *= subgroupNBasis[currN];
      currN--;
      continue;
    }
  }

  // C matrix layout
  auto [m, n] = opInfo.getResultMNIndex();
  int64_t cRank = opInfo.getCRank();

  // Get the M and N dims w.r.t. the dimensions of the C matrix. cMDims and
  // cNDims are the M and N dimensions of the C matrix in the order they are
  // iterated over in the contraction.
  SmallVector<int64_t> cMDims = opInfo.outMDims;
  SmallVector<int64_t> cNDims = opInfo.outNDims;
  SmallVector<int64_t> cBatchSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupStrides(cRank, 0);
  for (auto [i, dim] : llvm::enumerate(cMDims)) {
    cBatchSizes[dim] = batchMSizes[i];
    cSubgroupSizes[dim] = subgroupMBasis[i];
    cSubgroupStrides[dim] = subgroupMStrides[i];
  }
  for (auto [i, dim] : llvm::enumerate(cNDims)) {
    cBatchSizes[dim] = batchNSizes[i];
    cSubgroupSizes[dim] = subgroupNBasis[i];
    cSubgroupStrides[dim] = subgroupNStrides[i];
  }

  IREE::VectorExt::NestedLayoutAttr cLayout = createNestedLayout(
      context, cRank, m, n,
      /*subgroupCount=*/cSubgroupSizes,
      /*subgroupStrides=*/cSubgroupStrides,
      /*batchCount=*/cBatchSizes,
      getSingleSubgroupLayout(mmaAttr, IREE::GPU::MMAFragment::Acc));
  LLVM_DEBUG({ llvm::dbgs() << "C layout: " << cLayout << "\n"; });

  // A matrix layout
  auto [afm, bfn] = opInfo.getOperandMNIndex();
  auto [afk, bfk] = opInfo.getOperandKIndex();

  int64_t aRank = opInfo.getARank();

  SmallVector<int64_t> aMDims = opInfo.lhsMDims;
  SmallVector<int64_t> aBatchSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupStrides(aRank, 0);
  for (auto [i, dim] : llvm::enumerate(aMDims)) {
    aBatchSizes[dim] = batchMSizes[i];
    aSubgroupSizes[dim] = subgroupMBasis[i];
    aSubgroupStrides[dim] = subgroupMStrides[i];
  }
  for (auto [kDim, lhsKDim] :
       llvm::zip_equal(opInfo.getKDims(), opInfo.lhsKDim)) {
    aBatchSizes[lhsKDim] = bounds[kDim];
  }
  aBatchSizes[afk] = bounds[opInfo.getKDims().back()] / intrinsicK;

  IREE::VectorExt::NestedLayoutAttr aLayout = createNestedLayout(
      context, aRank, afm, afk,
      /*subgroupCount=*/aSubgroupSizes,
      /*subgroupStrides=*/aSubgroupStrides,
      /*batchCount=*/aBatchSizes,
      getSingleSubgroupLayout(mmaAttr, IREE::GPU::MMAFragment::Lhs));
  LLVM_DEBUG({ llvm::dbgs() << "A layout: " << aLayout << "\n"; });

  int64_t bRank = opInfo.getBRank();

  SmallVector<int64_t> bNDims = opInfo.rhsNDims;
  SmallVector<int64_t> bBatchSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupStrides(bRank, 0);
  for (auto [i, dim] : llvm::enumerate(bNDims)) {
    bBatchSizes[dim] = batchNSizes[i];
    bSubgroupSizes[dim] = subgroupNBasis[i];
    bSubgroupStrides[dim] = subgroupNStrides[i];
  }
  for (auto [kDim, rhsKDim] :
       llvm::zip_equal(opInfo.getKDims(), opInfo.rhsKDim)) {
    bBatchSizes[rhsKDim] = bounds[kDim];
  }
  bBatchSizes[bfk] = bounds[opInfo.getKDims().back()] / intrinsicK;

  IREE::VectorExt::NestedLayoutAttr bLayout = createNestedLayout(
      context, bRank, bfk, bfn,
      /*subgroupCount=*/bSubgroupSizes,
      /*subgroupStrides=*/bSubgroupStrides,
      /*batchCount=*/bBatchSizes,
      getSingleSubgroupLayout(mmaAttr, IREE::GPU::MMAFragment::Rhs));
  LLVM_DEBUG({ llvm::dbgs() << "B layout: " << bLayout << "\n"; });

  std::tuple<VectorLayoutInterface, VectorLayoutInterface,
             VectorLayoutInterface>
      result = {aLayout, bLayout, cLayout};
  return result;
}

static LogicalResult setContractionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                          SmallVector<bool> promotedOperands,
                                          RewriterBase &rewriter,
                                          linalg::LinalgOp contract) {
  // TODO: Add SIMT fallback.
  if (!schedule) {
    return contract->emitError("missing mma schedule for contraction");
  }

  // This function should have only be called on a contraction op.
  assert(linalg::isaContractionOpInterface(contract) &&
         "cannot set contraction anchor on non contraction op");

  FailureOr<VectorContractOpInfo> opInfo =
      VectorContractOpInfo::inferFromIndexingMaps(
          contract.getIndexingMapsArray());
  assert(succeeded(opInfo) && "contraction should have been inferred");

  auto layouts = getContractionLayout(schedule, opInfo.value(), contract);
  if (failed(layouts)) {
    return contract->emitError("cannot get concrete layout for contraction");
  }

  auto [aLayout, bLayout, cLayout] = *layouts;
  Location loc = contract.getLoc();

  Value lhs = contract->getOperand(0);
  Value rhs = contract->getOperand(1);
  Value acc = contract->getOperand(2);

  // Set layouts for lhs, rhs and acc.
  rewriter.setInsertionPoint(contract);
  auto layoutedLhs =
      rewriter.create<ToLayoutOp>(loc, lhs, aLayout, schedule.getIntrinsic());
  auto layoutedRhs =
      rewriter.create<ToLayoutOp>(loc, rhs, bLayout, schedule.getIntrinsic());
  auto layoutedAcc =
      rewriter.create<ToLayoutOp>(loc, acc, cLayout, schedule.getIntrinsic());

  // Promote matmul lhs and rhs.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  if (promotedOperands[0]) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[1]) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[2]) {
    layoutedAcc.setSharedMemoryConversion(true);
  }

  contract->setOperand(0, layoutedLhs.getResult());
  contract->setOperand(1, layoutedRhs.getResult());
  contract->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(contract);
  auto toLayout = rewriter.create<ToLayoutOp>(loc, contract->getResult(0),
                                              cLayout, schedule.getIntrinsic());
  rewriter.replaceAllUsesExcept(contract->getResult(0), toLayout.getResult(),
                                toLayout);

  return success();
}

static LogicalResult setConvolutionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                          SmallVector<bool> promotedOperands,
                                          RewriterBase &rewriter,
                                          linalg::LinalgOp conv) {
  // TODO: Add SIMT fallback.
  if (!schedule) {
    return conv->emitError("missing mma schedule for convolution");
  }

  // This function should have only be called on a convolution op.
  FailureOr<linalg::ConvolutionDimensions> convDims =
      linalg::inferConvolutionDims(conv);
  assert(succeeded(convDims) &&
         "cannot set convolution anchor on non convolution op");

  // Only convs with unit filter dims can be directly converted to matmul.
  SmallVector<int64_t> shape = conv.getStaticLoopRanges();
  if (!llvm::all_of(convDims->filterLoop,
                    [&shape](unsigned dim) { return shape[dim] == 1; })) {
    return failure();
  }

  llvm::SmallBitVector filterDims(conv.getNumLoops(), false);
  for (unsigned idx : convDims->filterLoop) {
    filterDims.set(idx);
  }

  SmallVector<AffineMap> maps = conv.getIndexingMapsArray();
  for (AffineMap &map : maps) {
    map = projectDims(map, filterDims, /*compressDimsFlag=*/false);
  }

  FailureOr<VectorContractOpInfo> opInfo =
      VectorContractOpInfo::inferFromIndexingMaps(maps);
  assert(succeeded(opInfo) &&
         "unit filter dim convolution should have been infered");

  auto layouts = getContractionLayout(schedule, opInfo.value(), conv);
  if (failed(layouts)) {
    return conv->emitError("cannot get concrete layout for convolution");
  }

  auto [aLayout, bLayout, cLayout] = *layouts;
  Location loc = conv.getLoc();

  Value lhs = conv->getOperand(0);
  Value rhs = conv->getOperand(1);
  Value acc = conv->getOperand(2);

  // Set layouts for lhs, rhs and acc.
  rewriter.setInsertionPoint(conv);
  auto layoutedLhs =
      rewriter.create<ToLayoutOp>(loc, lhs, aLayout, schedule.getIntrinsic());
  auto layoutedRhs =
      rewriter.create<ToLayoutOp>(loc, rhs, bLayout, schedule.getIntrinsic());
  auto layoutedAcc =
      rewriter.create<ToLayoutOp>(loc, acc, cLayout, schedule.getIntrinsic());

  // Promote matmul lhs and rhs.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  if (promotedOperands[0]) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[1]) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[2]) {
    layoutedAcc.setSharedMemoryConversion(true);
  }

  conv->setOperand(0, layoutedLhs.getResult());
  conv->setOperand(1, layoutedRhs.getResult());
  conv->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(conv);
  auto toLayout = rewriter.create<ToLayoutOp>(loc, conv->getResult(0), cLayout,
                                              schedule.getIntrinsic());
  rewriter.replaceAllUsesExcept(conv->getResult(0), toLayout.getResult(),
                                toLayout);

  return success();
}

/// Let's assume we have an matmul intrinsic (@) doing a matmul
/// ((M, K) X (K, N)) which produces a particular layout:
///
/// C = A @ B
///
/// If we transpose and swap the operands, we can keep the same matmul
/// intrinsic, but transpose the layout of the output intrinsic:
///
/// A.T = transpose(A)
/// B.T = transpose(B)
/// C.T = B.T @ A.T
/// C = transpose(C.T)
///
/// This is useful when the "@" instruction that the hardware lowers to
/// has a specific thread layout but the further uses of C expects a transposed
/// layout to the produced layout.
///
/// For example, for "@" lowering to AMDGPU MFMA instructions, the operands
/// have layout L and L.T and the result has the layout L.T .
/// So if you have a chain of matmuls:
///
/// C (L.T) = A (L) @ B (L.T)
/// E (L.T) = C (L.T)  @ D (L.T)
///            ^^^^^^^
///            Expected layout by instruction is L
///
/// To fix this, we can apply this transformation on the first matrix:
///
/// C.T (L.T) = B.T (L) @ A (L.T)
/// C   (L)   = transpose C.T (L.T)
/// E   (L.T) = C (L)  @ D (L.T)
///            ^^^^^
///            Layout matches the instruction!
///
/// Note that the mathematical formula
///   C = A @ B --> C.T = B.T @ A.T
/// is only defined on standard "@" function, it may be a different
/// transformation for other indexing maps.
///
/// For linalg operands, since the indexing maps are part of the op defination,
/// we can achieve the same transformation by simply swapping the operands.
static void swapOperandsToTransposeIntrinsic(RewriterBase &rewriter,
                                             linalg::GenericOp contractOp) {
  Value lhs = contractOp->getOperand(0);
  Value rhs = contractOp->getOperand(1);

  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  std::swap(indexingMaps[0], indexingMaps[1]);

  contractOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(indexingMaps));
  contractOp->setOperand(0, rhs);
  contractOp->setOperand(1, lhs);
}

static IREE::GPU::MMAScheduleAttr
transposeSchedule(RewriterBase &rewriter, IREE::GPU::MMAScheduleAttr schedule) {
  return rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
      schedule.getIntrinsic(), schedule.getSubgroupNCount(),
      schedule.getSubgroupMCount());
}

static LogicalResult setAttentionMatmulAnchor(RewriterBase &rewriter,
                                              linalg::LinalgOp qkMatmul,
                                              linalg::LinalgOp pvMatmul) {

  IREE::GPU::MMAScheduleAttr qkSchedule =
      rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(getIntrinsic(qkMatmul),
                                                   getSubgroupMCount(qkMatmul),
                                                   getSubgroupNCount(qkMatmul));

  IREE::GPU::MMAScheduleAttr pvSchedule =
      rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(getIntrinsic(pvMatmul),
                                                   getSubgroupMCount(pvMatmul),
                                                   getSubgroupNCount(pvMatmul));

  // Check if the intrinsic output for qkMatmul can be reused for pvMatmul.
  // We know that pvMatmul takes result of qkMatmul as it's lhs.
  // If the intrinsic output of pvMatmul can be used as rhs of pvMatmul,
  // we swap operands of both contracts to get output as transposed intrinsic.
  bool reuseIntrinsicOutput = false;
  bool transposeIntrinsic = false;

  auto qkIntrinsic =
      cast<IREE::GPU::MmaInterfaceAttr>(qkSchedule.getIntrinsic());
  auto pvIntrinsic =
      cast<IREE::GPU::MmaInterfaceAttr>(pvSchedule.getIntrinsic());
  IREE::GPU::MMASingleSubgroupLayout lhsLayout =
      getSingleSubgroupLayout(pvIntrinsic, IREE::GPU::MMAFragment::Lhs);
  IREE::GPU::MMASingleSubgroupLayout rhsLayout =
      getSingleSubgroupLayout(pvIntrinsic, IREE::GPU::MMAFragment::Rhs);
  IREE::GPU::MMASingleSubgroupLayout outLayout =
      getSingleSubgroupLayout(qkIntrinsic, IREE::GPU::MMAFragment::Acc);

  auto matchLayout = [](IREE::GPU::MMASingleSubgroupLayout layoutA,
                        IREE::GPU::MMASingleSubgroupLayout layoutB) -> bool {
    return (layoutA.element == layoutB.element) &&
           (layoutA.thread == layoutB.thread) &&
           (layoutA.tstrides == layoutB.tstrides);
  };

  // TODO: Move this check to KernelConfig and set appropriate attributes
  // in lowering_config for the operation. This allows us to check shared
  // memory usage and decide what kind of pipelining we can do.
  if (matchLayout(outLayout, lhsLayout)) {
    reuseIntrinsicOutput = true;
  } else if (matchLayout(outLayout, rhsLayout)) {
    reuseIntrinsicOutput = true;
    transposeIntrinsic = true;
  }

  SmallVector<bool> promotedQKOperands = getPromotedOperands(qkMatmul);
  SmallVector<bool> promotedPVOperands = getPromotedOperands(pvMatmul);

  // Do not promote lhs of pvMatmul if we are reusing the intrinsic output.
  promotedPVOperands[0] = !reuseIntrinsicOutput;

  // Transpose the intrinsic if requested. See docs for
  // swapOperandsToTransposeIntrinsic for more information on why this is done.
  if (transposeIntrinsic) {
    auto qkGeneric = dyn_cast<linalg::GenericOp>(qkMatmul.getOperation());
    auto pvGeneric = dyn_cast<linalg::GenericOp>(pvMatmul.getOperation());
    if (!qkGeneric || !pvGeneric) {
      pvMatmul->emitOpError("Non generic qkMatmul/pvMatmul transpose intrinsic "
                            "not yet implemented");
      return failure();
    }
    swapOperandsToTransposeIntrinsic(rewriter, qkGeneric);
    swapOperandsToTransposeIntrinsic(rewriter, pvGeneric);
    qkSchedule = transposeSchedule(rewriter, qkSchedule);
    pvSchedule = transposeSchedule(rewriter, pvSchedule);

    // Swap promoted operands.
    std::swap(promotedQKOperands[0], promotedQKOperands[1]);
    std::swap(promotedPVOperands[0], promotedPVOperands[1]);
  }

  if (failed(setContractionAnchor(qkSchedule, promotedQKOperands, rewriter,
                                  qkMatmul))) {
    return failure();
  }

  return setContractionAnchor(pvSchedule, promotedPVOperands, rewriter,
                              pvMatmul);
}

// Apply the permuted projection map to the layout.
static IREE::VectorExt::VectorLayoutInterface
getLayoutForMap(VectorLayoutInterface layout, AffineMap map) {
  // Project out unusued dims in layout.
  SmallVector<bool> projectedDims(layout.getRank(), false);
  for (int dim : getUnusedDimsBitVector(map).set_bits()) {
    projectedDims[dim] = true;
  }
  IREE::VectorExt::VectorLayoutInterface projectedLayout =
      layout.project(projectedDims);

  // Transpose dims in layout.
  AffineMap permMap = compressUnusedDims(map);
  SmallVector<int64_t> identity =
      llvm::to_vector(llvm::seq<int64_t>(permMap.getNumDims()));
  SmallVector<int64_t> perm = applyPermutationMap<int64_t>(permMap, identity);
  return projectedLayout.permute(perm);
}

static LogicalResult setDerivedThreadConfigLayout(
    IREE::GPU::DerivedThreadConfigAttr config, linalg::LinalgOp linalgOp,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  int64_t opRank = linalgOp.getNumLoops();

  SmallVector<int64_t> elementTile = config.getStaticTilingLevelSizes(
      static_cast<unsigned>(IREE::GPU::TilingLevel::Thread), linalgOp);

  SmallVector<int64_t> opShape = linalgOp.getStaticLoopRanges();
  for (auto [index, size, element] : llvm::enumerate(opShape, elementTile)) {
    if (ShapedType::isDynamic(size)) {
      linalgOp->emitError() << "Cannot set layouts for dynamic loop ranges";
      return failure();
    }

    if (size % element != 0) {
      linalgOp->emitError()
          << "Operation with unsupported number of elements. "
             "Chosen vector tile sizes for operation are not "
             "divisible by operation loop ranges at dim: "
          << index << ", size=" << size << ", vector size = " << element;
      return failure();
    }

    size /= element;
  }

  SmallVector<int64_t> threadTile(opRank, 1);
  SmallVector<int64_t> threadStrides(opRank, 0);

  int64_t residualThreads = ShapedType::getNumElements(workgroupSize);
  int64_t currStride = 1;

  for (auto [tile, stride, size] :
       llvm::reverse(llvm::zip(threadTile, threadStrides, opShape))) {
    int64_t threadBlock;
    if (residualThreads % size == 0) {
      threadBlock = size;
    } else if (size % residualThreads == 0) {
      threadBlock = residualThreads;
    } else {
      linalgOp->emitError() << "Operation with unsupported number of elements.";
      return failure();
    }

    tile = threadBlock;
    stride = currStride;
    size /= threadBlock;

    currStride *= threadBlock;
    residualThreads /= threadBlock;
  }

  SmallVector<int64_t> subgroupTile(opRank, 1);
  SmallVector<int64_t> subgroupStrides(opRank, 0);
  SmallVector<int64_t> outerTile(opRank, 1);

  MLIRContext *context = rewriter.getContext();
  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupTile, opShape, outerTile, threadTile, elementTile,
      subgroupStrides, threadStrides);

  Location loc = linalgOp.getLoc();

  rewriter.setInsertionPointAfter(linalgOp);
  for (OpResult result : linalgOp->getResults()) {
    VectorLayoutInterface resultLayout =
        getLayoutForMap(layout, linalgOp.getIndexingMapMatchingResult(result));
    auto toLayout = rewriter.create<ToLayoutOp>(loc, result, resultLayout);
    rewriter.replaceAllUsesExcept(result, toLayout, toLayout);
  }

  return success();
}

static LogicalResult setIntrinsicLoweringConfigLayout(
    IREE::GPU::LoweringConfigAttr config, linalg::LinalgOp candidate,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  SmallVector<bool> promotedOperands = getPromotedOperands(candidate);
  auto schedule = rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
      getIntrinsic(candidate), getSubgroupMCount(candidate),
      getSubgroupNCount(candidate));

  if (linalg::isaContractionOpInterface(candidate)) {
    if (succeeded(setContractionAnchor(schedule, promotedOperands, rewriter,
                                       candidate))) {
      return success();
    }
  }

  if (succeeded(linalg::inferConvolutionDims(candidate))) {
    if (succeeded(setConvolutionAnchor(schedule, promotedOperands, rewriter,
                                       candidate))) {
      return success();
    }
  }

  candidate->emitError() << "Unable to set intrinsic layouts on operation "
                            "based on given lowering config: "
                         << config;
  return failure();
}

static Operation *getOpWithAttr(Operation *root, StringRef attr) {
  Operation *result = nullptr;
  WalkResult walkResult = root->walk([&](Operation *op) {
    if (op->hasAttr(attr)) {
      if (result) {
        return WalkResult::interrupt();
      }
      result = op;
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return nullptr;
  }
  return result;
}

struct LLVMGPUConfigureTensorLayoutsPass final
    : impl::LLVMGPUConfigureTensorLayoutsPassBase<
          LLVMGPUConfigureTensorLayoutsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface func = getOperation();
    IRRewriter rewriter(func);

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(func);
    if (!maybeWorkgroupSize) {
      func->emitOpError()
          << "unable to query workgroup_size information from entry point";
      return signalPassFailure();
    }

    if (failed(setLayoutsFromLoweringConfig(func, maybeWorkgroupSize.value(),
                                            rewriter))) {
      return signalPassFailure();
    }

    auto attentionQKMatmul = dyn_cast_or_null<linalg::LinalgOp>(
        getOpWithAttr(func, "attention_qk_matmul"));
    auto attentionPVMatmul = dyn_cast_or_null<linalg::LinalgOp>(
        getOpWithAttr(func, "attention_pv_matmul"));

    if (attentionQKMatmul && !attentionPVMatmul) {
      func->emitError("Expected attention attributes to be set properly");
      return signalPassFailure();
    }

    if (!attentionQKMatmul && attentionPVMatmul) {
      func->emitError("Expected attention attributes to be set properly");
      return signalPassFailure();
    }

    if (attentionQKMatmul && attentionPVMatmul) {
      if (failed(setAttentionMatmulAnchor(rewriter, attentionQKMatmul,
                                          attentionPVMatmul))) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult setLayoutsFromLoweringConfig(FunctionOpInterface funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             RewriterBase &rewriter) {
    SmallVector<linalg::LinalgOp> candidates;
    funcOp->walk([&](linalg::LinalgOp op) {
      if (getLoweringConfig(op)) {
        candidates.push_back(op);
      }
    });

    for (linalg::LinalgOp candidate : candidates) {
      // Skip attention candidates.
      if (candidate->hasAttr("attention_qk_matmul") ||
          candidate->hasAttr("attention_pv_matmul")) {
        continue;
      }

      auto result =
          TypeSwitch<IREE::Codegen::LoweringConfigAttrInterface, LogicalResult>(
              getLoweringConfig(candidate))
              .Case([&](IREE::GPU::DerivedThreadConfigAttr config) {
                return setDerivedThreadConfigLayout(config, candidate,
                                                    workgroupSize, rewriter);
              })
              .Case([&](IREE::GPU::LoweringConfigAttr config) {
                if (config.getMmaKind()) {
                  return setIntrinsicLoweringConfigLayout(
                      config, candidate, workgroupSize, rewriter);
                }
                candidate->emitError() << "Unable to set layouts on operation "
                                          "based on given lowering config: "
                                       << config;
                return failure();
              })
              .Default([](Attribute) -> LogicalResult { return failure(); });

      if (failed(result)) {
        return failure();
      }
    }

    return success();
  }
};
} // namespace

} // namespace mlir::iree_compiler
