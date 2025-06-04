// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// InnerTiledOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> InnerTiledOp::getLoopIteratorTypes() {
  return getIteratorTypesArray();
}

SmallVector<Range> InnerTiledOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  SmallVector<AffineMap> indexingMaps = getIndexingMapsArray();
  for (const auto &it : llvm::enumerate(getIteratorTypes())) {
    // Search input map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), builder.getContext());
    auto iteratorType =
        llvm::cast<linalg::IteratorTypeAttr>(it.value()).getValue();
    if (iteratorType == utils::IteratorType::reduction) {
      OpFoldResult ub;
      for (auto [map, op] : llvm::zip_equal(indexingMaps, getOperands())) {
        // Get reduction dim size from first applicable shape, since they should
        // all match.
        std::optional<int64_t> dimIndex = map.getResultPosition(targetExpr);
        if (!dimIndex) {
          continue;
        }
        ub = tensor::getMixedSize(builder, loc, op, *dimIndex);
        ranges.emplace_back(Range{zero, ub, one});
        break;
      }
      assert(ub && "Reduction dimension must appera in some map");
      continue;
    }
    // Get parallel dimension size from result shape.
    std::optional<int64_t> resDimIndex =
        indexingMaps.back().getResultPosition(targetExpr);
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

static tensor::ExtractSliceOp extractSlice(OpBuilder &b, Location loc,
                                           Value src,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes,
                                           AffineMap indexingMap) {
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
InnerTiledOp::getTiledImplementation(OpBuilder &builder,
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
  SmallVector<Operation *> slices;

  for (auto [map, operand] : llvm::zip_equal(indexingMaps, getOperands())) {
    Operation *slice = extractSlice(builder, loc, operand, offsets, sizes, map);
    if (!slice) {
      return emitOpError("failed to get operand slice");
    }
    tiledOperands.emplace_back(slice->getResult(0));
    slices.push_back(slice);
  }
  SmallVector<Type, 4> resultTypes;
  resultTypes.push_back(tiledOperands.back().getType());

  Operation *tiledMmaOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledMmaOp}, SmallVector<Value>(tiledMmaOp->getResults()), slices};
}

LogicalResult InnerTiledOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  assert(resultNumber == 0);
  if (!hasTensorSemantics()) {
    return failure();
  }

  AffineMap resultMap = getIndexingMapsArray().back();
  if (resultMap.getNumDims() != offsets.size() ||
      offsets.size() != sizes.size()) {
    return failure();
  }

  populateSliceIndices(builder, getLoc(), getAcc(), offsets, sizes,
                       resultOffsets, resultSizes, resultMap);
  return success();
}

} // namespace mlir::iree_compiler::IREE::Codegen
