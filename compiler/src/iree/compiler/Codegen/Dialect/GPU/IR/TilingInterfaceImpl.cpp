// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// MultiMmaOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MultiMmaOp::getLoopIteratorTypes() {
  return getIteratorTypesArray();
}

SmallVector<Range> MultiMmaOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  SmallVector<AffineMap> indexingMaps = getIndexingMapsArray();
  for (const auto &it : llvm::enumerate(getIteratorTypes())) {
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), builder.getContext());
    auto iteratorType = llvm::cast<IteratorTypeAttr>(it.value()).getValue();
    if (iteratorType == utils::IteratorType::reduction) {
      // Get reduction dim size from lhs shape (same size in rhsShape).
      std::optional<int64_t> lhsDimIndex =
          indexingMaps[0].getResultPosition(targetExpr);
      assert(lhsDimIndex && "invalid lhs map");
      OpFoldResult ub =
          tensor::getMixedSize(builder, loc, getLhs(), *lhsDimIndex);
      ranges.emplace_back(Range{zero, ub, one});
      continue;
    }
    // Get parallel dimension size from result shape.
    std::optional<int64_t> resDimIndex =
        indexingMaps[2].getResultPosition(targetExpr);
    assert(resDimIndex && "invalid result map");
    OpFoldResult ub =
        tensor::getMixedSize(builder, loc, getAcc(), *resDimIndex);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

static void populateSliceIndices(OpBuilder &b, Location loc, Value src,
                                 ArrayRef<OpFoldResult> offsets,
                                 ArrayRef<OpFoldResult> sizes,
                                 SmallVector<OpFoldResult> &resultOffsets,
                                 SmallVector<OpFoldResult> &resultSizes,
                                 AffineMap indexingMap) {
  assert(offsets.size() == indexingMap.getNumDims() &&
         offsets.size() == sizes.size() && "Invalid tile");

  int64_t srcRank = cast<RankedTensorType>(src.getType()).getRank();

  OpFoldResult zero = b.getIndexAttr(0);
  resultOffsets.resize(srcRank, zero);
  resultSizes.resize(srcRank, zero);

  /// Populate the outer offset indices from the iteration space slice.
  for (auto [idx, dim] : llvm::enumerate(indexingMap.getResults())) {
    int64_t dimPos = cast<AffineDimExpr>(dim).getPosition();
    resultOffsets[idx] = offsets[dimPos];
    resultSizes[idx] = sizes[dimPos];
  }

  /// Populate the inner dim sizes based on the shape of the operand.
  for (int64_t i = indexingMap.getNumResults(), e = srcRank; i < e; ++i) {
    resultSizes[i] = tensor::getMixedSize(b, loc, src, i);
  }
}

static Value extractSlice(OpBuilder &b, Location loc, Value src,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes, AffineMap indexingMap) {
  assert(offsets.size() == indexingMap.getNumDims() &&
         offsets.size() == sizes.size() && "Invalid tile");

  int64_t srcRank = cast<RankedTensorType>(src.getType()).getRank();

  OpFoldResult zero = b.getIndexAttr(0);
  SmallVector<OpFoldResult> fullOffsets(srcRank, zero);
  SmallVector<OpFoldResult> fullSizes(srcRank, zero);
  populateSliceIndices(b, loc, src, offsets, sizes, fullOffsets, fullSizes,
                       indexingMap);

  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<OpFoldResult> fullStrides(srcRank, one);
  return b.create<tensor::ExtractSliceOp>(loc, src, fullOffsets, fullSizes,
                                          fullStrides);
}

FailureOr<TilingResult>
MultiMmaOp::getTiledImplementation(OpBuilder &builder,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes) {
  if (!hasTensorSemantics()) {
    return failure();
  }

  SmallVector<AffineMap, 4> indexingMaps = getIndexingMapsArray();
  if (offsets.size() != indexingMaps[0].getNumDims() ||
      offsets.size() != sizes.size()) {
    return failure();
  }

  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      extractSlice(builder, loc, getLhs(), offsets, sizes, indexingMaps[0]));
  tiledOperands.emplace_back(
      extractSlice(builder, loc, getRhs(), offsets, sizes, indexingMaps[1]));
  tiledOperands.emplace_back(
      extractSlice(builder, loc, getAcc(), offsets, sizes, indexingMaps[2]));

  SmallVector<Type, 4> resultTypes;
  resultTypes.push_back(tiledOperands.back().getType());

  Operation *tiledMmaOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledMmaOp},
                      SmallVector<Value>(tiledMmaOp->getResults())};
}

LogicalResult MultiMmaOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  assert(resultNumber == 0);
  if (!hasTensorSemantics()) {
    return failure();
  }

  AffineMap resultMap = getIndexingMapsArray()[2];
  if (resultMap.getNumDims() != offsets.size() ||
      offsets.size() != sizes.size()) {
    return failure();
  }

  populateSliceIndices(builder, getLoc(), getAcc(), offsets, sizes,
                       resultOffsets, resultSizes, resultMap);
  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
