// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-tiled-op-interface"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.cpp.inc"

/// Converts an `OpFoldResult` to a `Value` by building a constant op if
/// if the `OpFoldResult` is an `IntegerAttr`.
static Value getValue(OpBuilder &builder, Location loc,
                      OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return builder.create<ConstantIndexOp>(loc,
                                           attr.cast<IntegerAttr>().getInt());
  }
  return valueOrAttr.get<Value>();
}

//===----------------------------------------------------------------------===//
// Interface implementations for external operations.
//===----------------------------------------------------------------------===//

namespace {

/// External model for `tensor.extract_slice`.
struct ExtractSliceTiledOpInterface
    : public TiledOpInterface::ExternalModel<ExtractSliceTiledOpInterface,
                                             tensor::ExtractSliceOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    // No operand of `tensor.extract_slice` serves as a destination operand. So
    // create an `init_tensor` op of the same size as the result.
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    SmallVector<Value> dest;
    ReifiedRankedShapedTypeDims returnShape;
    (void)extractSliceOp.reifyResultShapes(b, returnShape);
    auto ofrShape = llvm::to_vector<4>(llvm::map_range(
        returnShape[0], [](Value v) { return getAsOpFoldResult(v); }));
    Value initTensor = b.create<linalg::InitTensorOp>(
        op->getLoc(), ofrShape, extractSliceOp.getType().getElementType());
    return {initTensor};
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    return SmallVector<StringRef>(extractSliceOp.getType().getRank(),
                                  getParallelIteratorTypeName());
  }

  SmallVector<Range> getLoopBounds(Operation *op, OpBuilder &b) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    SmallVector<Value> dest;
    ReifiedRankedShapedTypeDims returnShape;
    (void)extractSliceOp.reifyResultShapes(b, returnShape);
    Location loc = op->getLoc();
    Value zero = b.create<ConstantIndexOp>(loc, 0);
    Value one = b.create<ConstantIndexOp>(loc, 1);
    SmallVector<Range> loopRanges(returnShape[0].size(),
                                  Range{zero, nullptr, one});
    for (auto ub : enumerate(returnShape[0])) {
      loopRanges[ub.index()].size = ub.value();
    }
    return loopRanges;
  }

  Operation *getTiledImplementation(Operation *op, OpBuilder &b,
                                    ValueRange outputs,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &results) const {
    auto extractOp = cast<tensor::ExtractSliceOp>(op);
    // Check that strides are 1. For now abort if they arent.
    auto opStrides = extractOp.getMixedStrides();
    if (!llvm::all_of(opStrides, [&](OpFoldResult valueOrAttr) {
          Optional<int64_t> intVal = getConstantIntValue(valueOrAttr);
          return intVal && *intVal == 1;
        })) {
      op->emitOpError("unable to tile operation with non-unit stride");
      return nullptr;
    }
    Location loc = extractOp.getLoc();

    // Compute the offset and sizes for the tiled `tensor.extract_slice`
    // operation.
    llvm::SmallDenseSet<unsigned> droppedDims = extractOp.getDroppedDims();
    unsigned resultDimPos = 0;
    auto opOffsets = extractOp.getMixedOffsets();
    auto opSizes = extractOp.getMixedSizes();
    MLIRContext *context = b.getContext();
    SmallVector<OpFoldResult> newOffset, newSizes;
    for (auto opOffset : enumerate(opOffsets)) {
      // If the dimension is dropped, use the same offset.
      if (droppedDims.count(opOffset.index())) {
        newOffset.push_back(opOffset.value());
        newSizes.push_back(opSizes[opOffset.index()]);
      } else {
        AffineExpr d0, s0;
        bindDims(context, d0);
        bindSymbols(context, s0);
        AffineMap map = AffineMap::get(1, 1, d0 + s0);
        SmallVector<Value> operands = {getValue(b, loc, offsets[resultDimPos]),
                                       getValue(b, loc, opOffset.value())};
        Value offset = b.create<AffineApplyOp>(loc, map, operands);
        newOffset.push_back(offset);
        newSizes.push_back(sizes[resultDimPos]);
        resultDimPos++;
      }
    }
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> newStrides(opOffsets.size(), oneAttr);

    // Generate the tiled `tensor.extract_slice` operation.
    Type resultType = tensor::ExtractSliceOp::inferRankReducedResultType(
        extractOp.getType().getRank(), extractOp.getSourceType(), newOffset,
        newSizes, newStrides);
    auto tiledExtractOp = b.create<tensor::ExtractSliceOp>(
        loc, resultType.cast<RankedTensorType>(), extractOp.source(), newOffset,
        newSizes, newStrides);

    // Insert the tiled extract into the result tensor.
    SmallVector<OpFoldResult> resultStrides(offsets.size(), oneAttr);
    auto tiledInsertOp = b.create<tensor::InsertSliceOp>(
        loc, tiledExtractOp.result(), outputs[0], offsets, sizes,
        resultStrides);
    results.push_back(tiledInsertOp.result());
    return tiledExtractOp;
  }
};

struct InsertSliceTiledOpInterface
    : public TiledOpInterface::ExternalModel<InsertSliceTiledOpInterface,
                                             tensor::InsertSliceOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    SmallVector<Value> dest;
    dest.push_back(cast<tensor::InsertSliceOp>(op).dest());
    return dest;
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return SmallVector<StringRef>(insertSliceOp.getSourceType().getRank(),
                                  getParallelIteratorTypeName());
  }

  SmallVector<Range> getLoopBounds(Operation *op, OpBuilder &b) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    Value source = insertSliceOp.source();
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    Location loc = op->getLoc();
    Value zero = b.create<ConstantIndexOp>(loc, 0);
    Value one = b.create<ConstantIndexOp>(loc, 1);
    SmallVector<Range> loopBounds(sourceType.getRank(),
                                  Range{zero, nullptr, one});
    for (auto dim :
         llvm::seq<int64_t>(0, insertSliceOp.getSourceType().getRank())) {
      loopBounds[dim].size = b.create<tensor::DimOp>(loc, source, dim);
    }
    return loopBounds;
  }

  Operation *getTiledImplementation(Operation *op, OpBuilder &b,
                                    ValueRange outputs,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &results) const {
    auto insertOp = cast<tensor::InsertSliceOp>(op);
    // Compute a subtensor of the source based on the offsets.
    auto opStrides = insertOp.getMixedStrides();
    if (!llvm::all_of(opStrides, [&](OpFoldResult valueOrAttr) {
          Optional<int64_t> intVal = getConstantIntValue(valueOrAttr);
          return intVal && *intVal == 1;
        })) {
      op->emitOpError("unable to tile operation with non-unit stride");
      return nullptr;
    }
    MLIRContext *context = b.getContext();
    Location loc = insertOp.getLoc();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);
    auto extractSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, insertOp.source(), offsets, sizes, strides);

    // The offsets for the insert is based on the op offsets plus the offsets of
    // the loops passed in.
    auto opOffsets = insertOp.getMixedOffsets();
    auto opSizes = insertOp.getMixedSizes();
    unsigned offsetIndex = 0;
    ArrayRef<int64_t> sourceShape = insertOp.getSourceType().getShape();
    int64_t destRank = insertOp.getType().getRank();
    SmallVector<OpFoldResult> resultOffsets(destRank);
    SmallVector<OpFoldResult> resultSizes(destRank);
    for (auto opOffset : llvm::enumerate(opOffsets)) {
      // Check for rank-reducing by checking that
      // 1) The corresponding opSize value is 1
      // 2) The current rank of the source is not 1.
      // Then the opOffset is for the rank-reduced dimension. Skip.
      unsigned opOffsetIndex = opOffset.index();
      OpFoldResult opOffsetVal = opOffset.value();
      Optional<int64_t> opSizeVal = getConstantIntValue(opSizes[opOffsetIndex]);
      if (offsetIndex >= sourceShape.size() ||
          (opSizeVal && *opSizeVal == 1 && sourceShape[offsetIndex] != 1)) {
        resultOffsets[opOffsetIndex] = opOffsetVal;
        resultSizes[opOffsetIndex] = oneAttr;
        continue;
      }
      OpFoldResult offset = offsets[offsetIndex];
      if (opOffsetVal.is<Attribute>() && offset.is<Attribute>()) {
        resultOffsets[opOffsetIndex] = b.getI64IntegerAttr(
            *getConstantIntValue(opOffsetVal) + *getConstantIntValue(offset));
      } else {
        AffineExpr d0, s0;
        bindDims(context, d0);
        bindSymbols(context, s0);
        AffineMap map = AffineMap::get(1, 1, d0 + s0);
        SmallVector<Value> operands = {getValue(b, loc, offset),
                                       getValue(b, loc, opOffsetVal)};
        resultOffsets[opOffsetIndex] =
            b.create<AffineApplyOp>(loc, map, operands).getResult();
      }
      resultSizes[opOffsetIndex] = sizes[offsetIndex];
      offsetIndex++;
    }
    SmallVector<OpFoldResult> resultStrides(destRank, oneAttr);
    auto tiledInsertOp = b.create<tensor::InsertSliceOp>(
        loc, extractSliceOp.result(), outputs[0], resultOffsets, resultSizes,
        resultStrides);
    results.push_back(tiledInsertOp.result());
    return extractSliceOp;
  }
};
}  // namespace

void registerTiledOpInterfaceExternalModels(DialectRegistry &registry) {
  LLVM_DEBUG(
      { llvm::dbgs() << "Adding external models of tiled op interface\n"; });
  registry
      .addOpInterface<tensor::ExtractSliceOp, ExtractSliceTiledOpInterface>();
  registry.addOpInterface<tensor::InsertSliceOp, InsertSliceTiledOpInterface>();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir
