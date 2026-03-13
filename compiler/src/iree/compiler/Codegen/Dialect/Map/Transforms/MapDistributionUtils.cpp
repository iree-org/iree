// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapDistributionUtils.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

using IREE::Map::filterLeafInfos;
using IREE::Map::getLeafInfos;
using IREE::Map::LeafInfo;
using IREE::Map::PackLayoutAttr;

SmallVector<LeafDimInfo> getLeafDimMap(PackLayoutAttr layout) {
  SmallVector<LeafDimInfo> map;
  int64_t rank = layout.getRank();
  for (int64_t d = 0; d < rank; ++d) {
    SmallVector<LeafInfo> valLeaves =
        filterLeafInfos(layout.getShapeMode(d), layout.getStrideMode(d),
                        [](const LeafInfo &l) { return l.stride == 0; });
    if (valLeaves.empty()) {
      map.push_back({d, 1});
      continue;
    }
    for (const LeafInfo &leaf : valLeaves) {
      map.push_back({d, leaf.dataStride});
    }
  }
  return map;
}

/// Compute per-dimension thread offsets from a linearized thread ID.
///
/// Each dimension (mode) of a PackLayoutAttr has interleaved thread leaves
/// (stride > 0) and broadcast leaves (stride == 0). The goal is to convert a
/// flat thread ID into a data-space offset for each dimension.
///
/// Example: layout <((4, 2), (4, 8)) : ((1, 0), (0, 4))>
///   Dim 0 mode: sizes (4, 2), strides (1, 0) -> thread leaf size=4, bcast
///   Dim 1 mode: sizes (4, 8), strides (0, 4) -> bcast, thread leaf size=8
///   Result: offset_0 = (tid % 4) * 2,  offset_1 = (tid / 4) % 8
///
/// The algorithm has two phases:
///
/// Phase 1 (Delinearize): extract per-leaf coordinates from the thread ID.
///
///   Collect all thread leaves across all dimensions and use
///   basisFromSizesStrides to build a single delinearization basis from their
///   (size, threadStride) pairs. One affine.delinearize_index op then extracts
///   every thread-leaf coordinate at once.
///
/// Phase 2 (Linearize): convert coordinates to data-space offsets per dim.
///
///   Each dimension is handled independently. For a given mode with leaf sizes
///   [S0, S1, ..., Sn], we build a linearize_index where thread leaves use
///   their delinearized coordinate and broadcast leaves use constant 0.
///
///   This works because linearize_index computes:
///     result = coord_0 * (S1*S2*...*Sn) + coord_1 * (S2*...*Sn) + ... +
///     coord_n
///
///   The effective stride for position i is product(S_{i+1}...S_n), which is
///   exactly how getLeafInfos defines dataStride[i]. Broadcast zeros drop out,
///   so only thread leaves contribute: each scaled by its correct data stride.
///
///   For the example above, dim 0 produces:
///     linearize_index [coord, 0] by (4, 2) = coord * 2
///   which is the thread coordinate scaled by dataStride=2.
///
/// Why disjoint=true is valid on the linearize_index:
///
///   The disjoint flag asserts 0 <= coord_i < basis_i for every position.
///   - Thread coords: produced by delinearize, which computes (tid/s) % size,
///     guaranteeing the result is in [0, size). The linearize basis at that
///     position is the same leaf size, so the bound matches.
///   - Broadcast coords: constant 0, which is always < any leaf size (>= 1).
SmallVector<Value> buildThreadOffsets(OpBuilder &b, Location loc,
                                      PackLayoutAttr layout, Value threadId) {
  int64_t rank = layout.getRank();

  // Phase 1: Collect all thread leaves across all dimensions.
  SmallVector<int64_t> threadSizes, threadStrides;
  for (int64_t d = 0; d < rank; ++d) {
    for (const LeafInfo &leaf :
         filterLeafInfos(layout.getShapeMode(d), layout.getStrideMode(d),
                         [](const LeafInfo &l) { return l.stride > 0; })) {
      threadSizes.push_back(leaf.size);
      threadStrides.push_back(leaf.stride);
    }
  }

  // No thread leaves at all: every dim gets zero.
  if (threadSizes.empty()) {
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    return SmallVector<Value>(rank, zero);
  }

  // Delinearize threadId into per-leaf coordinates.
  SmallVector<int64_t> basis;
  SmallVector<size_t> dimToResult;
  [[maybe_unused]] LogicalResult res =
      basisFromSizesStrides(threadSizes, threadStrides, basis, dimToResult);
  // PackLayoutAttr's verifier guarantees thread strides are valid (injective,
  // non-overlapping), so basisFromSizesStrides must succeed here.
  assert(succeeded(res) && "thread strides must form a valid delinearization");

  auto delinOp = affine::AffineDelinearizeIndexOp::create(
      b, loc, threadId, basis, /*hasOuterBound=*/false);

  // Phase 2: Linearize per dimension using the mode's full leaf shape.
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> offsets;
  size_t entryIdx = 0;
  for (int64_t d = 0; d < rank; ++d) {
    SmallVector<LeafInfo> allLeaves =
        getLeafInfos(layout.getShapeMode(d), layout.getStrideMode(d));

    bool hasThreadLeaf = false;
    SmallVector<Value> coords;
    SmallVector<int64_t> leafSizes;
    for (const LeafInfo &leaf : allLeaves) {
      leafSizes.push_back(leaf.size);
      if (leaf.stride > 0) {
        hasThreadLeaf = true;
        coords.push_back(delinOp.getResult(dimToResult[entryIdx++]));
      } else {
        coords.push_back(zero);
      }
    }

    if (!hasThreadLeaf) {
      offsets.push_back(zero);
      continue;
    }
    offsets.push_back(affine::AffineLinearizeIndexOp::create(
        b, loc, coords, leafSizes, /*disjoint=*/true));
  }
  return offsets;
}

} // namespace mlir::iree_compiler
