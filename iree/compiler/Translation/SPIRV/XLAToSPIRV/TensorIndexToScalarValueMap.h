// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- TensorIndexToScalarValueMap.h ----------------------------*- C++ -*-===//
//
// Maintains mapping from Tensor value at an index to the scalar value during
// SPIR-V lowering.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_TENSORINDEXTOSCALAR_H_
#define IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_TENSORINDEXTOSCALAR_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

class TensorIndexToScalarValueMap {
 public:
  explicit TensorIndexToScalarValueMap() {}

  /// Returns the buffer associated with an argument in the dispatch function.
  Value getBufferForArgument(Value arg) { return argToBufferMap.lookup(arg); }

  /// Returns the value in the lowered function that represents the scalar value
  /// of the `tensor` in the original function at a given `index`
  Value getValueAtIndex(Value tensor, AffineMap index) {
    auto tensorIt = tensorIndexToScalarValueMap.find(tensor);
    if (tensorIt == tensorIndexToScalarValueMap.end()) {
      return nullptr;
    }
    auto scalarValIt = tensorIt->second.find(index);
    if (scalarValIt == tensorIt->second.end()) {
      return nullptr;
    }
    return scalarValIt->second;
  }

  Value getAccessIndicesForIndexMap(OpBuilder &builder, Location loc,
                                    const AffineMap &indexMap,
                                    ArrayRef<int64_t> shape = {1}) {
    assert(indexMap.getNumSymbols() <= symbolPosToValue.size() ||
           (symbolPosToValue.empty() && indexMap.getNumSymbols() == 0));
    // Linearize the shape using row major layout.
    // TODO(ravishankarm): Handle dynamic shapes here.
    SmallVector<int64_t, 2> strides(shape.size(), 1);
    AffineMap linearizedIndexMap;
    if (shape.size() > 1) {
      for (size_t dim = shape.size() - 1; dim > 0; --dim) {
        assert(shape[dim] != ShapedType::kDynamicSize);
        strides[dim - 1] = strides[dim] * shape[dim];
      }
      AffineMap linearizedMap = makeStridedLinearLayoutMap(
          strides, /*offset=*/0, builder.getContext());
      linearizedIndexMap = linearizedMap.compose(indexMap);
    } else {
      linearizedIndexMap = indexMap;
    }

    SmallVector<Value, 2> applyOperands;
    applyOperands.reserve(indexMap.getNumDims() + indexMap.getNumSymbols());
    for (auto i : llvm::seq<unsigned>(0, indexMap.getNumDims())) {
      auto dimValue = threadDimToValue.lookup(i);
      if (!dimValue) {
        emitError(loc, "unset value for dim d") << i;
        return nullptr;
      }
      applyOperands.push_back(dimValue);
    }
    for (auto i : llvm::seq<unsigned>(0, indexMap.getNumSymbols())) {
      auto symbolVal = symbolPosToValue.lookup(i);
      if (!symbolVal) {
        emitError(loc, "unset value for symbol s") << i;
        return nullptr;
      }
      applyOperands.push_back(symbolVal);
    }
    return builder.create<AffineApplyOp>(loc, linearizedIndexMap,
                                         applyOperands);
  }

  /// Records the `buffer` to use for an `argument` in the dispatch function.
  void setBufferForArgument(Value argument, Value buffer) {
    argToBufferMap[argument] = buffer;
  }

  /// Records the `scalar` value in the lowered function that represents the
  /// value of the `tensor` in the original function at a given `index`
  void setValueAtIndex(Value tensor, AffineMap index, Value scalar) {
    tensorIndexToScalarValueMap[tensor][index] = scalar;
  }

  /// Records the `value` to use for an AffineDimExpr while generating code for
  /// AffineExpr trees.
  void setDimValue(unsigned dim, const Value &value) {
    threadDimToValue[dim] = value;
  }

  /// Records the `value` to use for an AffineSymbolExpr while generating code
  /// for AffineExpr trees.
  void setSymbolValue(unsigned pos, Value value) {
    assert(!symbolPosToValue.count(pos));
    symbolPosToValue[pos] = value;
  }

 private:
  DenseMap<Value, DenseMap<AffineMap, Value>> tensorIndexToScalarValueMap;
  DenseMap<Value, Value> argToBufferMap;

  /// Mapping from AffineDimExpr to Value to use in generating scalar code to
  /// compute an AffineExpr tree.
  DenseMap<unsigned, Value> threadDimToValue;

  /// Mapping from AffineSymbolExpr to Value to use in generating scalar code
  /// to compute an AffineExpr tree.
  DenseMap<unsigned, Value> symbolPosToValue;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_TENSORINDEXTOSCALAR_H_
