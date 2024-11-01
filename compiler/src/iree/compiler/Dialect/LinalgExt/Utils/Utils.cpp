// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

OpFoldResult addOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto addMap = AffineMap::get(2, 0, {d0 + d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, addMap, {a, b});
}

OpFoldResult mulOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto addMap = AffineMap::get(2, 0, {d0 * d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, addMap, {a, b});
}

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.createOrFold<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.createOrFold<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getIndexAttr(t.getDimSize(dim));
}

SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc,
                                  Value shapedTypeValue) {
  return llvm::map_to_vector(
      llvm::seq<int64_t>(0,
                         cast<ShapedType>(shapedTypeValue.getType()).getRank()),
      [&](int64_t dim) { return getDim(builder, loc, shapedTypeValue, dim); });
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<Range> slice) {
  return getSlice(b, loc, src,
                  llvm::map_to_vector(slice, [](Range x) { return x.offset; }),
                  llvm::map_to_vector(slice, [](Range x) { return x.size; }),
                  llvm::map_to_vector(slice, [](Range x) { return x.stride; }));
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
                    ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(src.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return b.create<tensor::ExtractSliceOp>(loc, src, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
      })
      .Default([&](Type t) -> Operation * {
        assert(false && "invalid type");
        return nullptr;
      });
}

Value castValue(OpBuilder &b, Location loc, Value src, ShapedType type) {
  return TypeSwitch<Type, Value>(src.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        assert(isa<RankedTensorType>(type) && "expected compatible type");
        return b.create<tensor::CastOp>(loc, type, src)->getResult(0);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        assert(isa<MemRefType>(type) && "expected compatible type");
        return b.create<memref::CastOp>(loc, type, src)->getResult(0);
      })
      .Default([&](Type t) {
        assert(false && "invalid type");
        return nullptr;
      });
}

SmallVector<int64_t> computeInterchangeFromDimPos(ArrayRef<int64_t> dimsPos,
                                                  int64_t rank) {
  SmallVector<int64_t> interchangeVector;
  interchangeVector.reserve(dimsPos.size());
  // First map dims and their position. For example, dims_pos = [2, 0] will map
  // to:
  // [
  //  [ key: 2, value: 0]
  //  [ key: 0, value: 1]
  // ]
  // where key is the idx in dims_pos while value its position in dims_pos.
  DenseMap<int64_t, int64_t> dimsAndPosMapping;
  for (int64_t dimsIdx = 0, end = dimsPos.size(); dimsIdx < end; dimsIdx++) {
    dimsAndPosMapping[dimsPos[dimsIdx]] = dimsIdx;
  }

  // Scan the position in order and insert the value in the map
  // to compute the interchange vector.
  for (int64_t dimsIdx = 0; dimsIdx < rank; dimsIdx++) {
    if (dimsAndPosMapping.count(dimsIdx)) {
      interchangeVector.push_back(dimsAndPosMapping[dimsIdx]);
    }
  }
  return interchangeVector;
}

Value createValueFrom2DConstant(const float *val, int64_t rows, int64_t cols,
                                Location loc, RewriterBase &rewriter) {
  ArrayRef<float> vector(val, rows * cols);
  SmallVector<int64_t> shape{rows, cols};
  return rewriter.create<arith::ConstantOp>(
      loc, DenseFPElementsAttr::get(
               RankedTensorType::get(shape, rewriter.getF32Type()), vector));
}

SmallVector<int64_t> asShapeWithAnyValueAsDynamic(ArrayRef<OpFoldResult> ofrs) {
  SmallVector<int64_t> result;
  for (auto o : ofrs) {
    // Have to do this first, as getConstantIntValue special-cases constants.
    if (dyn_cast<Value>(o)) {
      result.push_back(ShapedType::kDynamic);
    } else {
      result.push_back(getConstantIntValue(o).value_or(ShapedType::kDynamic));
    }
  }
  return result;
}

SmallVector<AffineExpr> getDimExprsForSymbols(MLIRContext *context,
                                              unsigned numDims,
                                              unsigned numSymbols) {
  return llvm::map_to_vector(
      llvm::seq<unsigned>(0, numSymbols), [&](unsigned symbolNumber) {
        return getAffineDimExpr(symbolNumber + numDims, context);
      });
}

AffineMap convertDimsToSymbols(AffineMap map, unsigned numDims,
                               unsigned numSymbols,
                               SmallVector<AffineExpr> &symbolReplacements) {
  return map.replaceDimsAndSymbols(/*dimReplacements=*/ArrayRef<AffineExpr>{},
                                   symbolReplacements, numDims + numSymbols, 0);
}
SmallVector<AffineMap>
convertDimsToSymbols(ArrayRef<AffineMap> maps, unsigned numDims,
                     unsigned numSymbols,
                     SmallVector<AffineExpr> &symbolReplacements) {
  return llvm::map_to_vector(maps, [&](AffineMap map) {
    return convertDimsToSymbols(map, numDims, numSymbols, symbolReplacements);
  });
}
SmallVector<AffineMap> convertDimsToSymbols(MLIRContext *context,
                                            ArrayRef<AffineMap> maps,
                                            unsigned numDims,
                                            unsigned numSymbols) {
  auto symbolReplacements = getDimExprsForSymbols(context, numDims, numSymbols);
  return convertDimsToSymbols(maps, numDims, numSymbols, symbolReplacements);
}

//===---------------------------------------------------------------------===//
// Classification of ops that change bit-widths
//===---------------------------------------------------------------------===//

enum class BitWidthChangeInfo {
  kNull,
  kExtend,
  kTruncate,
};

static BitWidthChangeInfo isBitExtendOrTruncateOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return BitWidthChangeInfo::kNull;
  }

  if (genericOp.getNumDpsInits() != 1) {
    return BitWidthChangeInfo::kNull;
  }

  // Check that the all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return BitWidthChangeInfo::kNull;
  }

  // Check that all operands that have the highest rank have bit width
  // less than the output bit-width.
  DenseMap<int64_t, SmallVector<OpOperand *>> rankBuckets;
  int64_t maxOperandRank = 0;
  for (OpOperand *input : genericOp.getDpsInputOperands()) {
    auto inputType = dyn_cast<RankedTensorType>(input->get().getType());
    if (!inputType) {
      continue;
    }
    int64_t currRank = inputType.getRank();
    maxOperandRank = std::max(currRank, maxOperandRank);
    rankBuckets[currRank].push_back(input);
  }
  if (maxOperandRank == 0 || rankBuckets[maxOperandRank].empty()) {
    return BitWidthChangeInfo::kNull;
  }

  unsigned int maxInputElementBitWidth = 0;
  OpOperand *inputOperand;
  for (OpOperand *operand : rankBuckets[maxOperandRank]) {
    RankedTensorType tensorType =
        cast<RankedTensorType>(operand->get().getType());
    Type elementType = tensorType.getElementType();
    if (!elementType.isIntOrFloat()) {
      return BitWidthChangeInfo::kNull;
    }
    unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
    if (elementBitWidth > maxInputElementBitWidth) {
      maxInputElementBitWidth = elementBitWidth;
      inputOperand = operand;
    }
  }
  if (!inputOperand) {
    return BitWidthChangeInfo::kNull;
  }
  Type inputElementType =
      cast<RankedTensorType>(inputOperand->get().getType()).getElementType();

  // Check that the identity input element bitwidth is smaller than the output
  // element bitwidth.
  RankedTensorType outputType =
      dyn_cast<RankedTensorType>(genericOp->getResultTypes()[0]);
  if (!outputType) {
    return BitWidthChangeInfo::kNull;
  }
  Type outputElementType = outputType.getElementType();
  if (!outputElementType.isIntOrFloat()) {
    return BitWidthChangeInfo::kNull;
  }

  unsigned inputBitWidth = inputElementType.getIntOrFloatBitWidth();
  unsigned outputBitWidth = outputElementType.getIntOrFloatBitWidth();

  // Checks specific to bit extend operations.
  if (inputBitWidth < outputBitWidth) {
    // Since these are cloned into dispatches, avoid expensive operations.
    for (Operation &op : *genericOp.getBody()) {
      if (op.getDialect() == op.getContext()->getLoadedDialect("math")) {
        return BitWidthChangeInfo::kNull;
      }
    }
    return BitWidthChangeInfo::kExtend;
  }

  // Checks specific to bit truncate operations.
  if (outputBitWidth < inputBitWidth) {
    // For now enforce that the input and output ranks match for truncates.
    if (maxOperandRank != outputType.getRank()) {
      return BitWidthChangeInfo::kNull;
    }
    return BitWidthChangeInfo::kTruncate;
  }

  return BitWidthChangeInfo::kNull;
}

bool isBitExtendOp(Operation *op) {
  return isBitExtendOrTruncateOp(op) == BitWidthChangeInfo::kExtend;
}

bool isBitTruncateOp(Operation *op) {
  return isBitExtendOrTruncateOp(op) == BitWidthChangeInfo::kTruncate;
}

//===---------------------------------------------------------------------===//
// Classification of other ops
//===---------------------------------------------------------------------===//

bool isBroadcastingOp(linalg::LinalgOp op) {
  if (isa<linalg::BroadcastOp>(op)) {
    return true;
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp) {
    return false;
  }

  // Only allow a single input and init.
  if (genericOp.getNumDpsInits() != 1 || genericOp.getNumDpsInputs() != 1) {
    return false;
  }

  // Check that the all loops are parallel.
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return false;
  }

  // Check that indexing maps are broadcasting.
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  auto inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  auto outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (inMap.getNumResults() >= outMap.getNumResults()) {
    return false;
  }
  if (!inMap.isProjectedPermutation() || !outMap.isIdentity()) {
    return false;
  }
  return llvm::hasSingleElement(op.getBlock()->getOperations());
}

bool isGatherlikeOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return false;
  }

  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }

  auto &region = genericOp->getRegion(0);
  if (!llvm::hasSingleElement(region)) {
    return false;
  }

  // `yieldOp` should yield a single value from a `tensor.extract`
  auto yieldOp = cast<linalg::YieldOp>(region.front().getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return false;
  }
  auto extractOp = yieldOp.getOperand(0).getDefiningOp<tensor::ExtractOp>();
  if (!extractOp) {
    return false;
  }

  return true;
}

FailureOr<SmallVector<AffineMap>>
getIGEMMContractionIndexingMaps(linalg::LinalgOp linalgOp) {
  MLIRContext *ctx = linalgOp->getContext();
  return llvm::TypeSwitch<Operation *, FailureOr<SmallVector<AffineMap>>>(
             linalgOp.getOperation())
      .Case<linalg::Conv2DNchwFchwOp>(
          [&](linalg::Conv2DNchwFchwOp convOp) -> SmallVector<AffineMap> {
            AffineExpr bDim, mDim, nDim0, nDim1, kDim;
            bindDims(ctx, bDim, mDim, nDim0, nDim1, kDim);
            auto lhsMap = AffineMap::get(5, 0, {mDim, kDim}, ctx);
            auto rhsMap = AffineMap::get(5, 0, {bDim, nDim0, nDim1, kDim}, ctx);
            auto resultMap =
                AffineMap::get(5, 0, {bDim, mDim, nDim0, nDim1}, ctx);
            return {lhsMap, rhsMap, resultMap};
          })
      .Case<linalg::Conv2DNhwcHwcfOp>(
          [&](linalg::Conv2DNhwcHwcfOp convOp) -> SmallVector<AffineMap> {
            AffineExpr bDim, m0Dim, m1Dim, nDim, kDim;
            bindDims(ctx, bDim, m0Dim, m1Dim, nDim, kDim);
            auto lhsMap = AffineMap::get(5, 0, {bDim, m0Dim, m1Dim, kDim}, ctx);
            auto rhsMap = AffineMap::get(5, 0, {kDim, nDim}, ctx);
            auto resultMap =
                AffineMap::get(5, 0, {bDim, m0Dim, m1Dim, nDim}, ctx);
            return {lhsMap, rhsMap, resultMap};
          })
      .Default([](Operation *) { return failure(); });
}

FailureOr<SmallVector<int64_t>> getIGEMMLoopBounds(linalg::LinalgOp linalgOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<SmallVector<int64_t>>>(
             linalgOp.getOperation())
      .Case<linalg::Conv2DNchwFchwOp>(
          [&](linalg::Conv2DNchwFchwOp convOp) -> SmallVector<int64_t> {
            auto filterType =
                cast<RankedTensorType>(convOp.getOperandTypes()[1]);
            auto accType = cast<RankedTensorType>(convOp.getResultTypes()[0]);
            const int64_t B = accType.getDimSize(0);
            const int64_t N0 = accType.getDimSize(2);
            const int64_t N1 = accType.getDimSize(3);
            const int64_t M = filterType.getDimSize(0);
            const int64_t K = filterType.getDimSize(1) *
                              filterType.getDimSize(2) *
                              filterType.getDimSize(3);
            return {B, M, N0, N1, K};
          })
      .Case<linalg::Conv2DNhwcHwcfOp>(
          [&](linalg::Conv2DNhwcHwcfOp convOp) -> SmallVector<int64_t> {
            auto filterType =
                cast<RankedTensorType>(convOp.getOperandTypes()[1]);
            auto accType = cast<RankedTensorType>(convOp.getResultTypes()[0]);
            const int64_t B = accType.getDimSize(0);
            const int64_t M0 = accType.getDimSize(1);
            const int64_t M1 = accType.getDimSize(2);
            const int64_t N = accType.getDimSize(3);
            const int64_t K = filterType.getDimSize(0) *
                              filterType.getDimSize(1) *
                              filterType.getDimSize(2);
            return {B, M0, M1, N, K};
          })
      .Default([](Operation *) { return failure(); });
}

FailureOr<SmallVector<Value>> getIGEMMOperands(linalg::LinalgOp linalgOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<SmallVector<Value>>>(
             linalgOp.getOperation())
      .Case<linalg::Conv2DNchwFchwOp>(
          [&](linalg::Conv2DNchwFchwOp convOp) -> SmallVector<Value> {
            return {convOp.getOperands()[1], convOp.getOperands()[0],
                    convOp.getOperands()[2]};
          })
      .Case<linalg::Conv2DNhwcHwcfOp>(
          [&](linalg::Conv2DNhwcHwcfOp convOp) -> SmallVector<Value> {
            return {convOp.getOperands()[0], convOp.getOperands()[1],
                    convOp.getOperands()[2]};
          })
      .Default([](Operation *) { return failure(); });
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
