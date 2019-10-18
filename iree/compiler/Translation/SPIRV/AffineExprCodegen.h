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

//===- AffineExprCodegen.h -------------------------------------*- C++//-*-===//
//
// Code-generation for Affine Expression.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_AFFINEEXPRCODGEN_H
#define IREE_COMPILER_TRANSLATION_SPIRV_AFFINEEXPRCODGEN_H

#include "iree/compiler/Translation/SPIRV/XLAIndexPropagation.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/AffineExprVisitor.h"

namespace mlir {
namespace iree_compiler {

/// Codegenerator for affine expressions.
class AffineExprCodegen : public AffineExprVisitor<AffineExprCodegen, Value *> {
 public:
  explicit AffineExprCodegen(spirv::ModuleOp module,
                             IndexComputationCache &tensorIndices)
      : builder(module.getContext()),
        location(module.getLoc()),
        tensorIndices(tensorIndices) {}

  Value *visitAddExpr(AffineBinaryOpExpr expr) {
    auto operand1 = getValueInternal(expr.getLHS());
    auto operand2 = getValueInternal(expr.getRHS());
    return builder.create<spirv::IAddOp>(location, operand1, operand2);
  }
  Value *visitMulExpr(AffineBinaryOpExpr expr) {
    auto operand1 = getValueInternal(expr.getLHS());
    auto operand2 = getValueInternal(expr.getRHS());
    return builder.create<spirv::IMulOp>(location, operand1, operand2);
  }
  Value *visitModExpr(AffineBinaryOpExpr expr) {
    auto operand1 = getValueInternal(expr.getLHS());
    auto operand2 = getValueInternal(expr.getRHS());
    return builder.create<spirv::SModOp>(location, operand1, operand2);
  }
  Value *visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto operand1 = getValueInternal(expr.getLHS());
    auto operand2 = getValueInternal(expr.getRHS());
    return builder.create<spirv::SDivOp>(location, operand1, operand2);
  }
  Value *visitCeilDivExpr(AffineBinaryOpExpr expr) {
    // TODO(ravishankarm): Implement ceil div expr codegen.
    llvm_unreachable("Unimplemented affine AffineCeilDivExpr codegen");
    return nullptr;
  }
  Value *visitConstantExpr(AffineConstantExpr expr) {
    return builder.create<spirv::ConstantOp>(
        location, builder.getIntegerType(32),
        builder.getI32IntegerAttr(expr.getValue()));
  }
  Value *visitDimExpr(AffineDimExpr expr) {
    return threadDimToDstValue.lookup(expr.getPosition());
  }
  Value *visitSymbolExpr(AffineSymbolExpr expr) {
    // TODO(ravishankarm): Implement symbol expr codegen.
    llvm_unreachable("Unimplemented affine AffineSymbolExpr codegen");
    return nullptr;
  }

  /// Set the value that contains the workitem ID along a particular
  /// dimension. 0 -> x-dimension, 1 -> y-dimension, etc.
  void setDimDstValue(unsigned dimID, Value *value) {
    threadDimToDstValue[dimID] = value;
  }

  /// Generates the scalar value for a affine expression.
  Value *getValue(AffineExpr expr, OpBuilder::InsertPoint ip, Location loc) {
    auto &val = exprToDstValue[expr];
    if (!val) {
      location = loc;
      builder.restoreInsertionPoint(ip);
      val = visit(expr);
    }
    return val;
  }

  /// Returns a list of indices of a particular tensor in the source dialect
  /// needed within the dispatch function (obtained from the
  /// IndexComputationCache)
  SmallVector<AffineMap, 4> getIndices(Value *value) {
    SmallVector<AffineMap, 4> indices;
    for (auto &index : tensorIndices[value]) {
      indices.push_back(index.first);
    }
    return indices;
  }

  /// For a given tensor in the source dialect and index, return the index of
  /// all operands needed to compute the result.
  ArrayRef<AffineMap> getOperandIndices(Value *value, AffineMap index) {
    return tensorIndices[value][index];
  }

 private:
  /// Returns the Value corresponding to the AffineExpr `expr` by either
  /// previously generated value for the same index, or by generating the value.
  /// This version assumes the insertion point/Location has already been set.
  Value *getValueInternal(AffineExpr expr) {
    auto &val = exprToDstValue[expr];
    if (!val) {
      val = visit(expr);
    }
    return val;
  }

  OpBuilder builder;

  Location location;

  /// Map from launch dimension to scalar value.
  DenseMap<unsigned, Value *> threadDimToDstValue;

  /// Cache of affine expression to scalar value.  TODO(ravishankarm) : Might
  /// need to be changed if we are handling control flow within the dispatch
  /// function.
  DenseMap<AffineExpr, Value *> exprToDstValue;

  /// Map from tensor value in source dialect to list of indices of the tensor
  /// needed within a workitem to compute the results of the dispatch function.
  IndexComputationCache &tensorIndices;
};
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_AFFINEEXPRCODGEN_H
