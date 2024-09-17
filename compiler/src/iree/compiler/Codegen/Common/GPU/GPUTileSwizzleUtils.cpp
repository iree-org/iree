// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

namespace mlir::iree_compiler {

// Given an `expandShape` vector-of-vectors describing the mapping from source
// dimensions to expanded dimensions, returns the index of the first expanded
// dimension corresponding to the given source dimension index.
static int64_t
getExpandedDimFirstIdx(const SmallVector<SmallVector<int64_t>> &expandShape,
                       int64_t srcIndex) {
  int dstIndexFirst = 0;
  for (int i = 0; i < srcIndex; ++i) {
    dstIndexFirst += expandShape[i].size();
  }
  return dstIndexFirst;
}

void unroll(TileSwizzle &swizzle, int srcIndex, int unrollFactor) {
  assert(unrollFactor > 1);
  int dstIndexFirst = getExpandedDimFirstIdx(swizzle.expandShape, srcIndex);

  // The new unrolling dimension is inserted at the start of the expandShape
  // dimensions group corresponding to srcIndex.
  swizzle.expandShape[srcIndex].insert(swizzle.expandShape[srcIndex].begin(),
                                       unrollFactor);
  // Since we are not interleaving here, generating side-by-side copies of the
  // original layout, the new unrolling dimension is the new outermost
  // dimension. Existing entries get shifted to make room for it.
  for (auto &p : swizzle.permutation) {
    p += (p >= dstIndexFirst);
  }
  swizzle.permutation.insert(swizzle.permutation.begin(), dstIndexFirst);
}

void interleave(TileSwizzle &swizzle, int srcIndex,
                int expandedDimIndexToInterleaveAt) {
  // Compute which inner dimension to permute the current outer dimension into.
  int dstIndexFirst = getExpandedDimFirstIdx(swizzle.expandShape, srcIndex);
  int dstIndexToInterleaveAt = dstIndexFirst + expandedDimIndexToInterleaveAt;

  SmallVector<int64_t> outPermutation(swizzle.permutation.size());
  // The leading dimension, permutation[0], gets moved inwards to the
  // position that we just computed, dstIndexToInterleaveAt.
  outPermutation[dstIndexToInterleaveAt] = swizzle.permutation[0];
  // Outer dimensions get shifted outwards to fill the gap.
  for (int i = 0; i < dstIndexToInterleaveAt; ++i) {
    outPermutation[i] = swizzle.permutation[i + 1];
  }
  // Inner dimensions don't change. That is to say that we only interleave
  // at `targetInterleavedElements` granularity, we don't swizzle further
  // internally to that.
  for (int i = dstIndexToInterleaveAt + 1; i < outPermutation.size(); ++i) {
    outPermutation[i] = swizzle.permutation[i];
  }
  swizzle.permutation = outPermutation;
}

// Returns the permutation of indices that sorts `v` with the given comparator.
template <template <typename U> class Comparator, typename T>
static SmallVector<int64_t> getSortingPermutation(ArrayRef<T> v) {
  using P = std::pair<int64_t, T>;
  SmallVector<P> pairs;
  pairs.reserve(v.size());
  for (auto [i, x] : llvm::enumerate(v)) {
    pairs.push_back({i, x});
  }
  std::sort(pairs.begin(), pairs.end(),
            [](P p1, P p2) { return Comparator<T>{}(p1.second, p2.second); });
  SmallVector<int64_t> indices;
  for (auto p : pairs) {
    indices.push_back(p.first);
  }
  return indices;
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

  // Initially populate swizzle.expandShape with just the thread sizes, no
  // shape expansion for now.
  TileSwizzle swizzle;
  for (auto t : layout.thread) {
    swizzle.expandShape.push_back({t});
  }
  // The layout strides decide the initial swizzle.permutation.
  // Some WMMA intrinsics have tstrides=0 values, assert on that as that
  // would defeat this algorithm. We'll need to solve that if and when we want
  // to support data tiling on WMMA intrinsics.
  for (auto s : layout.tstrides) {
    (void)s;
    assert(s != 0);
  }
  swizzle.permutation =
      getSortingPermutation<std::greater, int64_t>(layout.tstrides);
  // Deal with any element size greater than 1 by inserting it innermost.
  // Notice that this is similar to the unroll() function, just creating an
  // inner dimension instead of an outer dimension.
  for (int i = 0; i < layout.element.size(); ++i) {
    if (layout.element[i] != 1) {
      swizzle.expandShape[i].push_back(layout.element[i]);
      int newIndex = getExpandedDimFirstIdx(swizzle.expandShape, i + 1) - 1;
      for (auto &p : swizzle.permutation) {
        p += (p >= newIndex);
      }
      swizzle.permutation.push_back(newIndex);
    }
  }
  // Deal with any outer size greater than 1 as just a call to unroll.
  // Iterate over dims in reverse order because we are creating a new outermost
  // dimension each time.
  for (int i = layout.outer.size() - 1; i >= 0; --i) {
    if (layout.outer[i] != 1) {
      unroll(swizzle, i, layout.outer[i]);
    }
  }

  return swizzle;
}

} // namespace mlir::iree_compiler
