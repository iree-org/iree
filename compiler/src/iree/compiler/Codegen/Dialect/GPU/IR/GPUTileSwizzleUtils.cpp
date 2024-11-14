// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
namespace mlir::iree_compiler {

using Kind = TileSwizzle::Dim::Kind;

// Returns the index of the first destination dimension corresponding to the
// given source dimension `srcIdx`.
static int64_t expandedDimIdx(const TileSwizzle::ExpandShapeType &expandShape,
                              int srcIdx) {
  int dstIdx = 0;
  for (int i = 0; i < srcIdx; ++i) {
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
static void expand(TileSwizzle &swizzle, int srcIdx, TileSwizzle::Dim dim) {
  int dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx);
  // The new unrolling dimension is inserted at the start of the expandShape
  // dimensions group corresponding to srcIdx.
  swizzle.expandShape[srcIdx].insert(swizzle.expandShape[srcIdx].begin(), dim);
  // Since we are not interleaving here, generating side-by-side copies of the
  // original layout, the new unrolling dimension is the new outermost
  // dimension. Existing entries get shifted to make room for it.
  for (auto &p : swizzle.permutation) {
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
static void interleave(TileSwizzle &swizzle, int srcIdx, int expandedIdx) {
  int dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx) + expandedIdx;
  SmallVector<int64_t> outPermutation(swizzle.permutation.size());
  // The leading dimension, permutation[0], gets moved inwards to the
  // position that we just computed, dstIdx.
  outPermutation[dstIdx] = swizzle.permutation[0];
  // Outer dimensions get shifted outwards to fill the gap.
  for (int i = 0; i < dstIdx; ++i) {
    outPermutation[i] = swizzle.permutation[i + 1];
  }
  // Inner dimensions don't change.
  for (int i = dstIdx + 1; i < outPermutation.size(); ++i) {
    outPermutation[i] = swizzle.permutation[i];
  }
  swizzle.permutation = outPermutation;
}

TileSwizzle getIntrinsicSwizzle(IREE::GPU::MMAIntrinsic intrinsic,
                                IREE::GPU::MMAFragment fragment) {
  auto layout = IREE::GPU::getSingleSubgroupLayout(intrinsic, fragment);

  // MMASingleSubgroupLayout has non-transposed RHS.
  // TileSwizzle has transposed RHS.
  if (fragment == IREE::GPU::MMAFragment::Rhs) {
    std::swap(layout.outer[0], layout.outer[1]);
    std::swap(layout.thread[0], layout.thread[1]);
    std::swap(layout.tstrides[0], layout.tstrides[1]);
    std::swap(layout.element[0], layout.element[1]);
  }

  TileSwizzle swizzle;
  // There are two source dimensions, corresponding to the arrays in `layout`
  // all having size 2. Let's just guard that assumption with one assert here.
  assert(layout.thread.size() == 2);
  swizzle.expandShape.resize(2);
  // Expand the shape from inner-most to outer-most dimension, so that we can
  // simply use the `expand` helper function, which creates new outer dims.
  // `layout.element` dims are inner-most, so we add them first.
  for (auto [i, e] : llvm::enumerate(layout.element)) {
    if (e != 1) {
      expand(swizzle, i, {Kind::Internal, e});
    }
  }
  // Next come `layout.thread` dims.
  for (auto [i, t] : llvm::enumerate(layout.thread)) {
    if (t != 1) {
      expand(swizzle, i, {Kind::CrossThread, t});
    }
  }
  // `layout.thread` dims are special in that they come with `layout.tstrides`
  // which may call for a swap in `swizzle.permutation`. We only need to worry
  // about that when both `layout.thread` sizes are greater than 1, so we didn't
  // skip them above. Note that this condition also implies that we don't need
  // to worry about `layout.tstrides == 0` which only happens with
  // `layout.thread == 1`.
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

static int getInnermostNonInternalDimIdx(
    const TileSwizzle::ExpandShapeDimVectorType &shape) {
  for (int idx = shape.size() - 1; idx >= 0; --idx) {
    if (shape[idx].kind != Kind::Internal) {
      return idx;
    }
  }
  assert(false && "all dimensions are internal!");
  return 0;
}

TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                       IREE::GPU::MMAFragment fragment) {
  auto swizzle = getIntrinsicSwizzle(mma.getIntrinsic().getValue(), fragment);
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs:
    // A-matrix (LHS). Source dimensions are M (index 0) and K (index 1).
    // Unroll on K with interleaving, then on M.
    if (mma.getUnrollK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getUnrollK()});
      int interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getUnrollM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getUnrollM()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
    break;
  case IREE::GPU::MMAFragment::Rhs:
    // B-matrix (RHS). Since the pack ops already took care of transposing B,
    // source dimensions are N (index 0) and K (index 1).
    // Unroll on K with interleaving, then on N.
    if (mma.getUnrollK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getUnrollK()});
      int interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getUnrollN() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getUnrollN()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    break;
  case IREE::GPU::MMAFragment::Acc:
    // C-matrix (accumulator). Source dimensions are M (index 0) and N (index
    // 1). Unroll on N, then on M.
    if (mma.getUnrollN() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getUnrollN()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 1, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    if (mma.getUnrollM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getUnrollM()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
    break;
  }
  return swizzle;
}

} // namespace mlir::iree_compiler
