// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-tiled-op-interface"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.cpp.inc"

/// Converts an `OpFoldResult` to a `Value` by building a constant op if
/// if the `OpFoldResult` is an `IntegerAttr`.
static Value getValue(OpBuilder &builder, Location loc,
                      OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return builder.create<arith::ConstantIndexOp>(
        loc, attr.cast<IntegerAttr>().getInt());
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

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    SmallVector<Value> dest;
    ReifiedRankedShapedTypeDims returnShape;
    (void)extractSliceOp.reifyResultShapes(b, returnShape);
    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
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
    // Check that strides are 1. For now abort if they arent
    Location loc = extractOp.getLoc();
    auto oneAttr = b.getI64IntegerAttr(1);

    // Compute the offset and sizes for the tiled `tensor.extract_slice`
    // operation.
    llvm::SmallBitVector droppedDims = extractOp.getDroppedDims();
    unsigned resultDimPos = 0;
    auto opOffsets = extractOp.getMixedOffsets();
    auto opSizes = extractOp.getMixedSizes();
    auto opStrides = extractOp.getMixedStrides();
    MLIRContext *context = b.getContext();
    SmallVector<OpFoldResult> newOffset, newSizes, newStrides;
    for (auto opOffset : enumerate(opOffsets)) {
      // If the dimension is dropped, use the same offset.
      if (droppedDims.test(opOffset.index())) {
        newOffset.push_back(opOffset.value());
        newSizes.push_back(opSizes[opOffset.index()]);
      } else {
        AffineExpr d0, s0, s1;
        bindDims(context, d0);
        bindSymbols(context, s0, s1);
        AffineMap map = AffineMap::get(1, 2, d0 * s0 + s1);
        SmallVector<Value> operands = {
            getValue(b, loc, offsets[resultDimPos]),
            getValue(b, loc, opStrides[opOffset.index()]),
            getValue(b, loc, opOffset.value())};
        Value offset = b.create<AffineApplyOp>(loc, map, operands);
        newOffset.push_back(offset);
        newSizes.push_back(sizes[resultDimPos]);
        resultDimPos++;
      }
      newStrides.push_back(opStrides[opOffset.index()]);
    }

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

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    Value source = insertSliceOp.source();
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
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

/// Forwards the implementation of `TiledOpInterface` to upstream
/// `TilingInterface`. Note that this forwarding is only valid when the
/// iteration space is same as the data space of the result(s). This is due to
/// the difference in the tiling algorithm being developed around
/// `TilingInterface` and that used with `TiledOpInterface`. The difference
/// comes down to the former only needing the tiled operation, and not the value
/// of the whole tensor.
template <typename OpTy>
struct ForwardToTilingInterface
    : public TiledOpInterface::ExternalModel<ForwardToTilingInterface<OpTy>,
                                             OpTy> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return cast<OpTy>(op).getDestinationOperands(b);
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    return cast<OpTy>(op).getLoopIteratorTypes();
  }
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    return cast<OpTy>(op).getIterationDomain(b);
  }
  Operation *getTiledImplementation(Operation *op, OpBuilder &b,
                                    ValueRange dest,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &results) const {
    SmallVector<Operation *> tiledOps = cast<OpTy>(op).getTiledImplementation(
        b, dest, offsets, sizes, /*tileDestOperands=*/true);
    if (tiledOps.empty()) {
      op->emitOpError("failed to tile operation");
      return nullptr;
    }
    assert(tiledOps.size() == 1 && "expected single tiled op");
    Operation *tiledOp = tiledOps.front();
    if (tiledOp->getNumResults() != dest.size()) {
      op->emitOpError(
          "mismatch in the number of results of the tiled operation and the "
          "number of results expected");
      return nullptr;
    }
    Location loc = op->getLoc();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);
    for (auto result : llvm::enumerate(tiledOp->getResults())) {
      // Assume that the shape of the result is same as the loop bounds of the
      // op. This implies the result can be inserted into the `dest` at
      // `offsets` and `sizes`. This would be illegal if that is not the
      // case. This is a point of difference between the `TiledOpInterface` in
      // IREE and `TilingInterface` in MLIR, since the latter sees fusion and
      // tiling as the same things. So it returns just the tiled op, and not the
      // result of the full tensor as the current tiling algorithm expects.
      auto tiledInsertOp = b.create<tensor::InsertSliceOp>(
          loc, result.value(), dest[result.index()], offsets, sizes, strides);
      results.push_back(tiledInsertOp);
    }
    return tiledOp;
  }
};

} // namespace

void IREE::LinalgExt::registerTiledOpInterfaceExternalModels(
    DialectRegistry &registry) {
  LLVM_DEBUG(
      { llvm::dbgs() << "Adding external models of tiled op interface\n"; });
  registry
      .addOpInterface<tensor::ExtractSliceOp, ExtractSliceTiledOpInterface>();
  registry.addOpInterface<tensor::InsertSliceOp, InsertSliceTiledOpInterface>();
  // TODO(ravishankarm): Needs custom PadTiledOpInterface or equiv.
  // registry.addOpInterface<tensor::PadOp,
  //                         ForwardToTilingInterface<tensor::PadOp>>();
}
