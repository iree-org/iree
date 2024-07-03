// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return builder.createOrFold<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
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

Value getSlice(OpBuilder &b, Location loc, Value src, ArrayRef<Range> slice) {
  return getSlice(b, loc, src,
                  llvm::map_to_vector(slice, [](Range x) { return x.offset; }),
                  llvm::map_to_vector(slice, [](Range x) { return x.size; }),
                  llvm::map_to_vector(slice, [](Range x) { return x.stride; }));
}

Value getSlice(OpBuilder &b, Location loc, Value src,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(src.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, src, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
      })
      .Default([&](Type t) {
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

} // namespace mlir::iree_compiler::IREE::LinalgExt
