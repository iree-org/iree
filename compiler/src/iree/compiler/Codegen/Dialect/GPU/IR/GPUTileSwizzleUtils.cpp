// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

#define DEBUG_TYPE "gpu-tile-swizzle-utils"

namespace mlir::iree_compiler::IREE::GPU {

using ::mlir::iree_compiler::IREE::Codegen::TileSwizzle;
using Kind = TileSwizzle::Dim::Kind;

SmallVector<int64_t>
sliceSwizzledShape(const TileSwizzle &swizzle,
                   llvm::function_ref<bool(TileSwizzle::Dim)> predicate) {
  SmallVector<int64_t> shape;
  for (TileSwizzle::ExpandShapeDimVectorType e : swizzle.expandShape) {
    for (TileSwizzle::Dim d : e) {
      shape.push_back(predicate(d) ? d.size : 1);
    }
  }
  applyPermutationToVector(shape, swizzle.permutation);
  return shape;
}

// Returns the index of the first destination dimension corresponding to the
// given source dimension `srcIdx`.
static size_t expandedDimIdx(const TileSwizzle::ExpandShapeType &expandShape,
                             size_t srcIdx) {
  size_t dstIdx = 0;
  for (size_t i = 0; i < srcIdx; ++i) {
    dstIdx += expandShape[i].size();
  }
  return dstIdx;
}

// Pushes `dim` to the front of `swizzle.expandShape[srcIdx]`, and updates
// `swizzle.permutation` to make the new dimension outer-most among the dims in
// `swizzle.expandShape[srcIdx]`.
//
// This can be used to unroll a kernel with kind = CrossIntrinsic,
// or to expand a kernel to multiple subgroups with kind = CrossThread.
//
// Example:
//    Input swizzle = { expandShape = [[16], [4]], permutation = [1, 0] }
//    Input srcIdx = 1
//    Input dim.size = 4
// -> Output swizzle = { expandShape = [[16], [4, 4]], permutation = [1, 2, 0] }
//
static void expand(TileSwizzle &swizzle, size_t srcIdx, TileSwizzle::Dim dim) {
  int64_t dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx);
  // The new unrolling dimension is inserted at the start of the expandShape
  // dimensions group corresponding to srcIdx.
  swizzle.expandShape[srcIdx].insert(swizzle.expandShape[srcIdx].begin(), dim);
  // Since we are not interleaving here, generating side-by-side copies of the
  // original layout, the new unrolling dimension is the new outermost
  // dimension. Existing entries get shifted to make room for it.
  for (int64_t &p : swizzle.permutation) {
    p += (p >= dstIdx);
  }
  swizzle.permutation.insert(swizzle.permutation.begin(), dstIdx);
}

/// Interleaves the layout in `swizzle` by mutating `swizzle.permutation` to
/// move permutation[0], the outer-most dimension (which the unroll() function
/// created to be the unrolling dimension), to the inner dimension given by
/// `dstIdx`.
///
/// Example:
///   Input swizzle = { expandShape = [[["CrossIntrinsic", 4],
///                                      ["CrossThread", 16]],
///                                     [["CrossIntrinsic", 2],
///                                      ["CrossThread", 4]]],
///                     permutation = [0, 3, 1, 2] }
///   Input dstIdx = 2
///   -> Output swizzle = { expandShape = [[["CrossIntrinsic", 4],
///                                          ["CrossThread", 16]],
///                                         [["CrossIntrinsic", 2],
///                                          ["CrossThread", 4]]],
///                         permutation = [3, 1, 0, 2] }
///
static void interleave(TileSwizzle &swizzle, size_t dstIdx) {
  assert(dstIdx < swizzle.permutation.size() && "dstIdx out of bounds");

  SmallVector<int64_t> outPermutation(swizzle.permutation.size());
  // The leading dimension, permutation[0], gets moved inwards to the
  // inner position, dstIdx.
  outPermutation[dstIdx] = swizzle.permutation[0];
  // Outer dimensions get shifted outwards to fill the gap.
  for (size_t i = 0; i < dstIdx; ++i) {
    outPermutation[i] = swizzle.permutation[i + 1];
  }
  // Inner dimensions don't change.
  for (size_t i = dstIdx + 1; i < outPermutation.size(); ++i) {
    outPermutation[i] = swizzle.permutation[i];
  }
  swizzle.permutation = outPermutation;
}

template <typename MMAIntrinsicTy>
static TileSwizzle getIntrinsicSwizzle(MMAIntrinsicTy intrinsic,
                                       unsigned operandIdx) {
  IREE::GPU::MMASingleSubgroupLayout layout =
      IREE::GPU::getSingleSubgroupLayout(intrinsic, operandIdx);
  const bool isScaled =
      std::is_same<MMAIntrinsicTy, IREE::GPU::ScaledMMAIntrinsic>::value;
  const bool isLhs = isIntrinsicLhs<MMAIntrinsicTy>(operandIdx);
  const bool isRhs = isIntrinsicRhs<MMAIntrinsicTy>(operandIdx);
  const bool isRhsScale = isIntrinsicRhsScale<MMAIntrinsicTy>(operandIdx);

  // MMASingleSubgroupLayout has non-transposed RHS and RHS scales, but
  // TileSwizzle has transposed RHS and RHS scales, so reorder the `layout`
  // to match the TileSwizzle.
  auto swapRHSKAndN = [](MutableArrayRef<int64_t> v) {
    // The RHS layout is [K, N], and the RHS scales layout is [K, Kb, N], so
    // rotate right by 1 element to swap [K, Kb] and N.
    std::rotate(v.begin(), v.end() - 1, v.end());
  };
  if (isRhs || isRhsScale) {
    swapRHSKAndN(layout.outer);
    swapRHSKAndN(layout.thread);
    swapRHSKAndN(layout.tstrides);
    swapRHSKAndN(layout.element);
  }

  TileSwizzle swizzle;
  // There are 3 source dimensions for LHS and RHS if the matmul is scaled.
  // All other operands (and LHS/RHS for non-scaled matmuls) have 2 source
  // dimensions. These correspond to the arrays in `layout` all having a
  // matching size. Let's just guard that assumption with one assert here.
  const unsigned numSrcDims = isScaled && (isLhs || isRhs) ? 3 : 2;
  assert(layout.thread.size() == numSrcDims &&
         "expected layout rank to match the number of source dims");
  swizzle.expandShape.resize(numSrcDims);
  // Expand the shape from inner-most to outer-most dimension, so that we can
  // simply use the `expand` helper function, which creates new outer dims.
  // `layout.element` dims are inner-most, so we add them first.
  // Iterate layout.element in reverse, because we always want the `Kb`
  // dimension to be innermost.
  for (auto [i, e] : llvm::enumerate(llvm::reverse(layout.element))) {
    if (e != 1) {
      size_t srcIdx = layout.element.size() - 1 - i;
      expand(swizzle, srcIdx, {Kind::Internal, e});
    }
  }
  // Next come `layout.thread` dims.
  int64_t subgroupSize = getIntrinsicSubgroupSize(intrinsic);
  int64_t numThreadsInLayout = llvm::product_of(layout.thread);
  assert(subgroupSize % numThreadsInLayout == 0 &&
         "expected subgroupSize to be divisible by numThreadsInLayout");
  assert(subgroupSize >= numThreadsInLayout &&
         "expected at most subgroupSize threads in the layout");
  int64_t extraDistributionFactor = subgroupSize / numThreadsInLayout;
  // Based on the MMA layouts, there is expected to be at most one dim with a
  // tstride of 0.
  assert(llvm::count(layout.tstrides, 0) <= 1 &&
         "expected at most one dim with a tstride of 0");
  for (auto [i, t, s] : llvm::enumerate(layout.thread, layout.tstrides)) {
    // If the thread has a stride of 0, then we need a dimension for it in the
    // swizzle so we can distribute by more than a factor of 1 along the dim.
    if (t != 1 || s == 0) {
      TileSwizzle::Dim tDim(Kind::CrossThread, t);
      if (s == 0) {
        tDim.distributionSize *= extraDistributionFactor;
      }
      expand(swizzle, i, tDim);
    }
  }
  // `layout.thread` dims are special in that they come with `layout.tstrides`
  // which may call for a swap in `swizzle.permutation`. We only need to worry
  // about that when both `layout.thread` sizes are greater than 1, so we didn't
  // skip them above. Note that this condition also implies that we don't need
  // to worry about `layout.tstrides == 0` which only happens with
  // `layout.thread == 1`.
  // The `thread` size for the `Kb` dimension is always 1 with a tstride of 1,
  // so there can only be 2 layout.thread dims to check here, and we can do a
  // simple swap instead of a rotate.
  assert((layout.thread.size() == 2 ||
          (layout.thread.size() == 3 && layout.thread[2] == 1)) &&
         "expected inner thread dim to be 1 for blocked LHS or RHS");
  if (layout.thread[0] != 1 && layout.thread[1] != 1 &&
      layout.tstrides[0] > layout.tstrides[1]) {
    std::swap(swizzle.permutation[0], swizzle.permutation[1]);
  }
  // Finally come `layout.outer` dims, added last so they are outer-most.
  for (auto [i, o] : llvm::enumerate(layout.outer)) {
    if (o != 1) {
      expand(swizzle, i, {Kind::Internal, o});
    }
  }
  return swizzle;
}

/// Returns the index of the innermost cross thread dimension after applying the
/// swizzle permutation.
///
/// Example:
///   Input swizzle = { expandShape = [[["CrossIntrinsic", 4],
///                                      ["CrossThread", 16]],
///                                     [["CrossIntrinsic", 2],
///                                      ["CrossThread", 4]]],
///                     permutation = [0, 3, 1, 2] }
///   -> flatDims = [["CrossIntrinsic", 4], ["CrossThread", 4],
///                  ["CrossThread", 16], ["CrossIntrinsic", 2]]
///   -> Output innermostCrossThreadDimIdx = 2
static size_t getInnermostCrossThreadDimIdx(const TileSwizzle &swizzle) {
  // Flatten the expandShape.
  SmallVector<TileSwizzle::Dim> flatDims;
  for (const auto &shape : swizzle.expandShape) {
    flatDims.append(shape.begin(), shape.end());
  }
  // Apply the permutation to the flatDims.
  applyPermutationToVector(flatDims, swizzle.permutation);
  // Iterate in reverse to find the innermost CrossThread dimension.
  for (int64_t i = flatDims.size() - 1; i >= 0; --i) {
    if (flatDims[i].kind == Kind::CrossThread) {
      return i;
    }
  }
  assert(false && "no cross thread dimension found!");
  return 0;
}

static void expandIfNonUnit(TileSwizzle &swizzle, size_t srcIdx,
                            TileSwizzle::Dim dim, bool interleave = false) {
  if (dim.size > 1) {
    expand(swizzle, srcIdx, dim);
    if (interleave) {
      IREE::GPU::interleave(swizzle, getInnermostCrossThreadDimIdx(swizzle));
    }
  }
}

/// Implementation of `getSwizzle` for both scaled and non-scaled matmuls.
template <typename MMAAttrTy>
static TileSwizzle getSwizzleImpl(MMAAttrTy mma, unsigned operandIdx) {
  TileSwizzle swizzle = getIntrinsicSwizzle(mma.getIntrinsic(), operandIdx);
  using MMAIntrinsicTy = decltype(mma.getIntrinsic());
  const bool isLhs = isIntrinsicLhs<MMAIntrinsicTy>(operandIdx);
  const bool isRhs = isIntrinsicRhs<MMAIntrinsicTy>(operandIdx);
  const bool isAcc = isIntrinsicAcc<MMAIntrinsicTy>(operandIdx);
  const bool isLhsScale = isIntrinsicLhsScale<MMAIntrinsicTy>(operandIdx);
  const bool isRhsScale = isIntrinsicRhsScale<MMAIntrinsicTy>(operandIdx);
  auto contains = [](DenseI64ArrayAttr attr, int64_t val) {
    return attr ? llvm::is_contained(attr.asArrayRef(), val) : false;
  };
  const bool interleaveM =
      contains(mma.getOperandsInterleavingIntrinsicsM(), operandIdx);
  const bool interleaveN =
      contains(mma.getOperandsInterleavingIntrinsicsN(), operandIdx);
  const bool interleaveK =
      contains(mma.getOperandsInterleavingIntrinsicsK(), operandIdx);
  TileSwizzle::Dim subgroupsM = {Kind::CrossThread, mma.getSubgroupsM()};
  TileSwizzle::Dim subgroupsN = {Kind::CrossThread, mma.getSubgroupsN()};
  TileSwizzle::Dim subgroupsK = {Kind::CrossThread, mma.getSubgroupsK()};
  TileSwizzle::Dim intrinsicsM = {Kind::CrossIntrinsic, mma.getIntrinsicsM()};
  TileSwizzle::Dim intrinsicsN = {Kind::CrossIntrinsic, mma.getIntrinsicsN()};
  TileSwizzle::Dim intrinsicsK = {Kind::CrossIntrinsic, mma.getIntrinsicsK()};
  if (isLhs || isLhsScale) {
    subgroupsM.distributionSize *= subgroupsN.size;
    constexpr int M = 0, K = 1;
    expandIfNonUnit(swizzle, K, intrinsicsK, interleaveK);
    expandIfNonUnit(swizzle, M, intrinsicsM, interleaveM);
    expandIfNonUnit(swizzle, K, subgroupsK);
    expandIfNonUnit(swizzle, M, subgroupsM);
  } else if (isRhs || isRhsScale) {
    constexpr int N = 0, K = 1;
    expandIfNonUnit(swizzle, K, intrinsicsK, interleaveK);
    expandIfNonUnit(swizzle, N, intrinsicsN, interleaveN);
    expandIfNonUnit(swizzle, K, subgroupsK);
    expandIfNonUnit(swizzle, N, subgroupsN);
  } else if (isAcc) {
    if (subgroupsN.size > 1) {
      subgroupsN.distributionSize *= subgroupsK.size;
    } else if (subgroupsM.size > 1) {
      subgroupsM.distributionSize *= subgroupsK.size;
    }
    constexpr int M = 0, N = 1;
    expandIfNonUnit(swizzle, N, intrinsicsN, interleaveN);
    expandIfNonUnit(swizzle, M, intrinsicsM, interleaveM);
    expandIfNonUnit(swizzle, N, subgroupsN);
    expandIfNonUnit(swizzle, M, subgroupsM);
  }
  return swizzle;
}

TileSwizzle getSwizzle(IREE::GPU::DataTiledScaledMMAAttr scaledMma,
                       unsigned operandIdx) {
  return getSwizzleImpl(scaledMma, operandIdx);
}

TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma, int operandIndex) {
  return getSwizzleImpl(mma, operandIndex);
}

/// Remove the expanded dimensions for this index and update the permutation
/// by erasing the removed dimensions' indices and adjusting existing larger
/// indices accordingly.
static void remove(TileSwizzle &swizzle, size_t idx) {
  assert(idx < swizzle.expandShape.size() && "idx out of bounds");
  const size_t startIdx = llvm::accumulate(
      ArrayRef(swizzle.expandShape).take_front(idx), size_t(0),
      [](size_t idx, const TileSwizzle::ExpandShapeDimVectorType &dims)
          -> size_t { return idx + dims.size(); });
  const size_t endIdx = startIdx + swizzle.expandShape[idx].size();
  swizzle.expandShape.erase(swizzle.expandShape.begin() + idx);
  SmallVector<int64_t> newPermutation;
  for (const int64_t &p : swizzle.permutation) {
    if (p < startIdx) {
      newPermutation.push_back(p);
    } else if (p >= endIdx) {
      newPermutation.push_back(p - (endIdx - startIdx));
    }
  }
  swizzle.permutation = newPermutation;
}

FailureOr<TileSwizzle>
getEncodingSwizzle(IREE::Encoding::EncodingAttr encoding,
                   IREE::GPU::DataTiledMMAInterfaceAttr mma,
                   unsigned operandIndex) {
  FailureOr<Codegen::EncodingContractionLikeDimInfo> dims =
      Codegen::getEncodingContractionLikeDims(encoding);
  if (failed(dims)) {
    return failure();
  }
  TileSwizzle swizzle = mma.getTileSwizzle(operandIndex);
  // Some dimensions may be missing from the encoding due to broadcasting.
  // Remove any dimensions that are supposed to be present for the given
  // operand, but are not present for the encoding.
  // The swizzle.expandShape is an expansion of the packed inner tiles.
  // The order of dimensions in the inner tiles follow the following
  // priorities:
  //  - LHS: [M, K, Kb]
  //  - RHS: [N, K, Kb]
  //  - LHS Scales: [M, Kb]
  //  - RHS Scales: [N, Kb]
  //  - ACC: [M, N]
  // The order of dimensions for all operands is therefore: [M, N, K, Kb].
  // Remove dimensions from outermost to innermost, so we can always just
  // remove the outer dimension regardless of the operand.
  if (dims->mDim.shouldHaveDim) {
    if (!dims->mDim.operandIdx.has_value()) {
      remove(swizzle, 0);
    }
  }
  if (dims->nDim.shouldHaveDim) {
    if (!dims->nDim.operandIdx.has_value()) {
      remove(swizzle, 0);
    }
  }
  if (dims->kDim.shouldHaveDim) {
    if (!dims->kDim.operandIdx.has_value()) {
      remove(swizzle, 0);
    }
  }
  if (dims->kBDim.shouldHaveDim) {
    if (!dims->kBDim.operandIdx.has_value()) {
      remove(swizzle, 0);
    }
  }
  return swizzle;
}

} // namespace mlir::iree_compiler::IREE::GPU
