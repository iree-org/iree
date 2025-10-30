// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

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

// Interleaves the layout in `swizzle` by mutating `swizzle.permutation` to
// move permutation[0], the outer-most dimension (which the unroll() function
// created to be the unrolling dimension), to the inner dimension given by
// `expandedIdx`.
//
// Example:
//    Input swizzle = { expandShape = [[16], [4, 4]], permutation = [1, 2, 0] }
//    Input srcIdx = 1
//    Input expandedIdx = 1
// -> Output swizzle = { expandShape = [[16], [4, 4]], permutation = [2, 0, 1] }
//
static void interleave(TileSwizzle &swizzle, size_t srcIdx, int expandedIdx) {
  size_t dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx) + expandedIdx;
  SmallVector<int64_t> outPermutation(swizzle.permutation.size());
  // The leading dimension, permutation[0], gets moved inwards to the
  // position that we just computed, dstIdx.
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
  IREE::GPU::MMASingleSubgroupLayout layout;
  const bool isScaled =
      std::is_same<MMAIntrinsicTy, IREE::GPU::ScaledMMAIntrinsic>::value;
  const unsigned lhsIdx = 0;
  const unsigned rhsIdx = 1;
  const unsigned lhsScalesIdx = 2;
  const unsigned rhsScalesIdx = 3;
  const bool isLHSorRHS = operandIdx == lhsIdx || operandIdx == rhsIdx;
  if (isScaled) {
    // The operand mapping for `getSingleSubgroupLayout` follows a different
    // operand order than is used for TileSwizzle, so we need to remap the
    // operandIdx to get the right layout. The layouts for TileSwizzle vs.
    // `getSingleSubgroupLayout` are shown below:
    //             | TileSwizzle | getSingleSubgroupLayout
    //         LHS | 0           | 0
    //         RHS | 1           | 2
    //  LHS Scales | 2           | 1
    //  RHS Scales | 3           | 3
    //         ACC | 4           | 4
    // TODO(Max191): Decide on a consistent operand order for both.
    int64_t layoutOperandIdx = operandIdx;
    if (operandIdx == rhsIdx) {
      layoutOperandIdx = 2;
    } else if (operandIdx == lhsScalesIdx) {
      layoutOperandIdx = 1;
    }
    layout = IREE::GPU::getSingleSubgroupLayout(
        static_cast<ScaledMMAIntrinsic>(intrinsic), layoutOperandIdx);
  } else {
    layout = IREE::GPU::getSingleSubgroupLayout(
        static_cast<MMAIntrinsic>(intrinsic),
        static_cast<IREE::GPU::MMAFragment>(operandIdx));
  }

  // MMASingleSubgroupLayout has non-transposed RHS and RHS scales, but
  // TileSwizzle has transposed RHS and RHS scales, so reorder the `layout`
  // to match the TileSwizzle.
  auto swapRHSKAndN = [](MutableArrayRef<int64_t> v) {
    // The RHS layout is [K, N], and the RHS scales layout is [K, Kb, N], so
    // rotate right by 1 element to swap [K, Kb] and N.
    std::rotate(v.begin(), v.end() - 1, v.end());
  };
  if (operandIdx == rhsIdx || (isScaled && operandIdx == rhsScalesIdx)) {
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
  const unsigned numSrcDims = isScaled && isLHSorRHS ? 3 : 2;
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

static size_t getInnermostNonInternalDimIdx(
    const TileSwizzle::ExpandShapeDimVectorType &shape) {
  for (size_t idx = shape.size() - 1; idx >= 0; --idx) {
    if (shape[idx].kind != Kind::Internal) {
      return idx;
    }
  }
  assert(false && "all dimensions are internal!");
  return 0;
}

/// Implementation of `getSwizzle` for both scaled and non-scaled matmuls.
template <typename MMAAttrTy>
static TileSwizzle getSwizzleImpl(MMAAttrTy mma, unsigned operandIdx) {
  TileSwizzle swizzle = getIntrinsicSwizzle(mma.getIntrinsic(), operandIdx);
  const bool isScaled =
      std::is_same<MMAAttrTy, IREE::GPU::DataTiledScaledMMAAttr>::value;
  const unsigned lhsIdx = 0;
  const unsigned rhsIdx = 1;
  const unsigned lhsScalesIdx = 2;
  const unsigned rhsScalesIdx = 3;
  const unsigned accIdx = isScaled ? 4 : 2;
  const bool isRhsScales = isScaled && operandIdx == rhsScalesIdx;
  const bool isLhsScales = isScaled && operandIdx == lhsScalesIdx;
  if (operandIdx == lhsIdx || isLhsScales) {
    // A-matrix (LHS). Source dimensions are M (index 0) and K (index 1).
    // Unroll on K with interleaving, then on M.
    if (mma.getIntrinsicsK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getIntrinsicsK()});
      size_t interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      // For scaled matmuls, interleaving happens because we want to load all
      // the unrolled scales with each vector load, so we need to interleave at
      // the very last dimension for the scales. For the LHS, we load in blocks,
      // so we don't need to interleave.
      if (isLhsScales) {
        interleavingIdx = swizzle.expandShape[1].size() - 1;
      }
      if (!isScaled || isLhsScales) {
        interleave(swizzle, 1, interleavingIdx);
      }
    }
    if (mma.getIntrinsicsM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getIntrinsicsM()});
    }
    if (mma.getSubgroupsM() > 1) {
      // Although subGroupsN is not part of the LHS swizzle, we must still
      // delinearize over the combined subGroupsM × subGroupsN space. By
      // contrast, RHS does *not* need special handling, since subGroupsM can be
      // treated as an implicit leading dimension and omitted anyway.
      TileSwizzle::Dim dim(Kind::CrossThread, mma.getSubgroupsM(),
                           mma.getSubgroupsM() * mma.getSubgroupsN());
      expand(swizzle, 0, dim);
    }
  } else if (operandIdx == rhsIdx || isRhsScales) {
    // B-matrix (RHS). Since the pack ops already took care of transposing B,
    // source dimensions are N (index 0) and K (index 1).
    // Unroll on K with interleaving, then on N.
    if (mma.getIntrinsicsK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getIntrinsicsK()});
      size_t interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      // Like with the LHS above, we want to interleave such that we load all
      // the unrolled scales with each vector load.
      if (isRhsScales) {
        interleavingIdx = swizzle.expandShape[1].size() - 1;
      }
      if (!isScaled || isRhsScales) {
        interleave(swizzle, 1, interleavingIdx);
      }
    }
    if (mma.getIntrinsicsN() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getIntrinsicsN()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsN()});
    }
  } else if (operandIdx == accIdx) {
    // C-matrix (accumulator). Source dimensions are M (index 0) and N (index
    // 1). Unroll on N, then on M.
    if (mma.getIntrinsicsN() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getIntrinsicsN()});
    }
    if (mma.getIntrinsicsM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getIntrinsicsM()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 1, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
  }
  return swizzle;
}

TileSwizzle getSwizzle(IREE::GPU::DataTiledScaledMMAAttr scaledMma,
                       unsigned operandIdx) {
  return getSwizzleImpl(scaledMma, operandIdx);
}

TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                       IREE::GPU::MMAFragment fragment) {
  return getSwizzleImpl(mma, static_cast<unsigned>(fragment));
}

/// Remove the expanded dimensions for this index and update the permutation by
/// erasing the removed dimensions' indices and adjusting existing larger
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
  // The order of dimensions in the inner tiles follow the following priorities:
  //  - LHS: [M, K, Kb]
  //  - RHS: [N, K, Kb]
  //  - LHS Scales: [M, Kb]
  //  - RHS Scales: [N, Kb]
  //  - ACC: [M, N]
  // The order of dimensions for all operands is therefore: [M, N, K, Kb].
  // Remove dimensions from outermost to innermost, so we can always just remove
  // the outer dimension regardless of the operand.
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
